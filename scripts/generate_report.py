"""
Script per la generazione automatica del report Quarto.
"""
import subprocess
import sys
import os
from pathlib import Path


def render_quarto_report(
    input_file="reports/report_lucy.qmd",
    output_format="all",
    output_dir=None,
    execute=True
):
    """
    Genera il report Quarto in uno o più formati.
    
    Args:
        input_file: File .qmd da renderizzare
        output_format: Formato output ('html', 'pdf', 'revealjs', 'all')
        output_dir: Directory di output (None = default)
        execute: Se True, esegue il codice Python
        
    Returns:
        bool: True se il rendering è riuscito
    """
    formats = {
        'html': 'html',
        'pdf': 'pdf',
        'revealjs': 'revealjs',
        'all': ['html', 'pdf', 'revealjs']
    }
    
    if output_format not in formats:
        print(f"Formato non supportato: {output_format}")
        return False
    
    target_formats = formats[output_format]
    if not isinstance(target_formats, list):
        target_formats = [target_formats]
    
    success = True
    for fmt in target_formats:
        print(f"\n{'='*60}")
        print(f"Rendering in formato: {fmt.upper()}")
        print(f"{'='*60}\n")
        
        cmd = ['quarto', 'render', input_file, '--to', fmt]
        
        if output_dir:
            cmd.extend(['--output-dir', output_dir])
        
        if not execute:
            cmd.append('--no-execute')
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            print(f"OK: Report {fmt} generato con successo!")
        except subprocess.CalledProcessError as e:
            print(f"ERRORE: Errore durante il rendering in {fmt}:")
            print(e.stderr)
            success = False
        except FileNotFoundError:
            print("ERRORE: Quarto non trovato. Assicurati che sia installato e nel PATH.")
            success = False
    
    return success


def main():
    """Funzione principale."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Genera report Quarto automaticamente"
    )
    parser.add_argument(
        '--format',
        choices=['html', 'pdf', 'revealjs', 'all'],
        default='html',
        help='Formato di output (default: html)'
    )
    parser.add_argument(
        '--input',
        default='reports/report_lucy.qmd',
        help='File .qmd da renderizzare (default: reports/report_lucy.qmd)'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory di output (opzionale)'
    )
    parser.add_argument(
        '--no-execute',
        action='store_true',
        help='Non eseguire il codice Python (usa cache)'
    )
    
    args = parser.parse_args()
    
    # Verifica che il file esista
    if not Path(args.input).exists():
        print(f"ERRORE: File non trovato: {args.input}")
        sys.exit(1)
    
    # Verifica variabile d'ambiente per AI
    if not os.getenv("OPENAI_API_KEY"):
        print("ATTENZIONE: OPENAI_API_KEY non configurata.")
        print("   Le funzionalità AI non saranno disponibili.")
        print("   Configura la variabile d'ambiente per abilitarle.\n")
    
    # Renderizza il report
    success = render_quarto_report(
        input_file=args.input,
        output_format=args.format,
        output_dir=args.output_dir,
        execute=not args.no_execute
    )
    
    if success:
        print("\n" + "="*60)
        print("OK: Generazione report completata!")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("ERRORE: Errore durante la generazione del report")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
