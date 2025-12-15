"""
Script per la generazione automatica del report Quarto.
"""
import subprocess
import sys
import os
import re
import time
from pathlib import Path
from typing import Optional, List, Union

# Carica variabili d'ambiente da file env (es. .env) prima di lanciare Quarto,
# così il processo `quarto render` eredita le API keys.
try:
    from dotenv import load_dotenv, find_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    for _name in [".env", "-env"]:
        _p = _PROJECT_ROOT / _name
        if _p.exists():
            load_dotenv(dotenv_path=_p, override=False)

    _found = find_dotenv(usecwd=True)
    if _found:
        load_dotenv(dotenv_path=_found, override=False)
except Exception:
    # Se python-dotenv non è disponibile o qualcosa va storto, continuiamo comunque:
    # il report può essere renderizzato anche senza AI.
    pass

# Costanti
SUPPORTED_FORMATS = ['html', 'pdf', 'revealjs', 'all']
DEFAULT_FORMAT = 'html'
DEFAULT_INPUT = 'reports/report_lucy.qmd'

# Pattern per estrarre informazioni sulle celle da Quarto
CELL_PATTERN = re.compile(r'Cell\s+(\d+)/(\d+):\s*(.+?)(?:\s+\.\.\.\s+Done)?$', re.IGNORECASE)


def format_progress_bar(current: int, total: int, label: str = "", width: int = 50) -> str:
    """
    Crea una barra di progresso testuale.
    
    Args:
        current: Valore corrente
        total: Valore totale
        label: Etichetta opzionale
        width: Larghezza della barra in caratteri
        
    Returns:
        Stringa formattata con la barra di progresso
    """
    if total == 0:
        percent = 0
    else:
        percent = min(100, int((current / total) * 100))
    
    filled = int((current / total) * width) if total > 0 else 0
    bar = '#' * filled + '-' * (width - filled)
    
    label_str = f" | {label}" if label else ""
    return f"[{bar}] {percent:3d}% ({current}/{total}){label_str}"


def update_progress(current: int, total: int, label: str = "", start_time: Optional[float] = None):
    """
    Aggiorna e mostra la barra di progresso.
    
    Args:
        current: Valore corrente
        total: Valore totale
        label: Etichetta opzionale
        start_time: Timestamp di inizio per calcolare tempo trascorso
    """
    bar = format_progress_bar(current, total, label)
    
    if start_time:
        elapsed = time.time() - start_time
        if current > 0 and total > 0:
            estimated_total = elapsed * total / current
            remaining = estimated_total - elapsed
            time_str = f" | Tempo: {int(elapsed)}s / ~{int(estimated_total)}s (rimanenti: ~{int(remaining)}s)"
        else:
            time_str = f" | Tempo: {int(elapsed)}s"
    else:
        time_str = ""
    
    # Usa \r per sovrascrivere la riga precedente
    print(f"\r{bar}{time_str}", end='', flush=True)
    
    # Se completato, vai a nuova riga
    if current >= total:
        print()


def render_quarto_report(
    input_file: str = DEFAULT_INPUT,
    output_format: str = DEFAULT_FORMAT,
    output_dir: Optional[str] = None,
    execute: bool = True
) -> bool:
    """
    Genera il report Quarto in uno o più formati.
    
    Args:
        input_file: File .qmd da renderizzare
        output_format: Formato output ('html', 'pdf', 'revealjs', 'all')
        output_dir: Directory di output (None = default)
        execute: Se True, esegue il codice Python
        
    Returns:
        True se il rendering è riuscito
    """
    formats = {
        'html': 'html',
        'pdf': 'pdf',
        'revealjs': 'revealjs',
        'all': ['html', 'pdf', 'revealjs']
    }
    
    if output_format not in formats:
        print(f"ERRORE: Formato non supportato: {output_format}")
        print(f"Formati supportati: {', '.join(SUPPORTED_FORMATS)}")
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
            # Usa Popen per leggere output in streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Variabili per tracciare il progresso
            current_cell = 0
            total_cells = 0
            current_label = ""
            start_time = time.time()
            output_lines = []
            
            # Leggi output in streaming
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                
                output_lines.append(line)
                
                # Cerca pattern "Cell X/Y: label"
                match = CELL_PATTERN.search(line)
                if match:
                    current_cell = int(match.group(1))
                    total_cells = int(match.group(2))
                    current_label = match.group(3).strip()
                    update_progress(current_cell, total_cells, current_label, start_time)
                else:
                    # Stampa altre righe importanti (errori, warning, etc.)
                    if any(keyword in line.lower() for keyword in ['error', 'warning', 'done', 'complete']):
                        # Vai a nuova riga per mostrare il messaggio
                        print()
                        print(line.rstrip())
                        # Riprendi la barra di progresso se abbiamo ancora celle
                        if total_cells > 0:
                            update_progress(current_cell, total_cells, current_label, start_time)
            
            # Attendi che il processo termini
            return_code = process.wait()
            
            # Mostra progresso finale
            if total_cells > 0:
                update_progress(total_cells, total_cells, "Completato", start_time)
            else:
                # Se non abbiamo trovato pattern di celle, mostra output completo
                print("\n" + "".join(output_lines))
            
            if return_code == 0:
                elapsed = time.time() - start_time
                print(f"\n[OK] Report {fmt} generato con successo! (Tempo totale: {int(elapsed)}s)")
            else:
                print(f"\n[ERRORE] Errore durante il rendering in {fmt} (codice: {return_code})")
                print("Output completo:")
                print("".join(output_lines))
                success = False
                
        except FileNotFoundError:
            print("ERRORE: Quarto non trovato. Assicurati che sia installato e nel PATH.")
            print("   Installa Quarto da: https://quarto.org/docs/get-started/")
            success = False
    
    return success


def main() -> None:
    """Funzione principale."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Genera report Quarto automaticamente"
    )
    parser.add_argument(
        '--format',
        choices=SUPPORTED_FORMATS,
        default=DEFAULT_FORMAT,
        help=f'Formato di output (default: {DEFAULT_FORMAT})'
    )
    parser.add_argument(
        '--input',
        default=DEFAULT_INPUT,
        help=f'File .qmd da renderizzare (default: {DEFAULT_INPUT})'
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
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERRORE: File non trovato: {args.input}")
        print(f"   Percorso assoluto cercato: {input_path.absolute()}")
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
