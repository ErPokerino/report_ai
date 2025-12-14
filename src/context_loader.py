"""
Modulo per il caricamento e la processazione di file di contesto dalla cartella context.
Supporta PDF e Markdown, estraendo sezioni rilevanti per il contesto del LLM.
"""
import os
import re
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Configurazione logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Se non ci sono handler, aggiungine uno
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Estrae il testo da un file PDF.
    
    Args:
        pdf_path: Percorso al file PDF
        
    Returns:
        str: Testo estratto dal PDF
    """
    if not PDF_AVAILABLE:
        return f"[ERRORE: pypdf non installato. Installa con: uv add pypdf]"
    
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                text_parts.append(f"--- Pagina {page_num} ---\n{text}")
        
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Errore nell'estrazione del PDF {pdf_path.name}: {e}")
        return f"[ERRORE nell'estrazione del PDF {pdf_path.name}: {str(e)}]"


def extract_text_from_markdown(md_path: Path) -> str:
    """
    Estrae il testo da un file Markdown.
    
    Args:
        md_path: Percorso al file Markdown
        
    Returns:
        str: Contenuto del file Markdown
    """
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Errore nella lettura del file Markdown {md_path.name}: {e}")
        return f"[ERRORE nella lettura del file {md_path.name}: {str(e)}]"


def split_into_sections(text: str, min_section_length: int = 100) -> List[Dict[str, str]]:
    """
    Divide il testo in sezioni basate su titoli/intestazioni.
    
    Args:
        text: Testo da dividere
        min_section_length: Lunghezza minima di una sezione
        
    Returns:
        List[Dict]: Lista di sezioni con 'title' e 'content'
    """
    sections = []
    
    # Pattern per identificare titoli (Markdown: #, ##, ###, etc. o linee con testo in maiuscolo)
    title_pattern = r'^(#{1,6}\s+.+?)$|^([A-Z][A-Z\s]{3,})$'
    
    lines = text.split('\n')
    current_section = {'title': 'Introduzione', 'content': []}
    
    for line in lines:
        # Controlla se è un titolo
        if re.match(title_pattern, line.strip()):
            # Salva la sezione precedente se ha contenuto sufficiente
            if len('\n'.join(current_section['content'])) >= min_section_length:
                sections.append({
                    'title': current_section['title'],
                    'content': '\n'.join(current_section['content']).strip()
                })
            
            # Inizia una nuova sezione
            title = line.strip().lstrip('#').strip()
            current_section = {'title': title, 'content': []}
        else:
            current_section['content'].append(line)
    
    # Aggiungi l'ultima sezione
    if len('\n'.join(current_section['content'])) >= min_section_length:
        sections.append({
            'title': current_section['title'],
            'content': '\n'.join(current_section['content']).strip()
        })
    
    return sections


def find_relevant_sections(sections: List[Dict[str, str]], 
                          keywords: List[str],
                          max_sections: int = 5) -> List[Dict[str, str]]:
    """
    Trova le sezioni più rilevanti basate su keyword.
    
    Args:
        sections: Lista di sezioni
        keywords: Lista di keyword da cercare
        max_sections: Numero massimo di sezioni da restituire
        
    Returns:
        List[Dict]: Sezioni più rilevanti ordinate per rilevanza
    """
    if not keywords:
        return sections[:max_sections]
    
    # Calcola score di rilevanza per ogni sezione
    scored_sections = []
    keywords_lower = [kw.lower() for kw in keywords]
    
    for section in sections:
        score = 0
        text_lower = (section['title'] + ' ' + section['content']).lower()
        
        for keyword in keywords_lower:
            # Punteggio più alto se la keyword è nel titolo
            if keyword in section['title'].lower():
                score += 3
            # Punteggio medio se è nel contenuto
            score += text_lower.count(keyword)
        
        scored_sections.append((score, section))
    
    # Ordina per score decrescente
    scored_sections.sort(key=lambda x: x[0], reverse=True)
    
    # Restituisci le top sezioni
    relevant = [section for score, section in scored_sections if score > 0]
    
    # Se non ci sono sezioni rilevanti, restituisci le prime
    if not relevant:
        relevant = sections[:max_sections]
    else:
        relevant = relevant[:max_sections]
    
    return relevant


def load_context_files(context_dir: str = "context",
                       keywords: Optional[List[str]] = None,
                       max_sections_per_file: int = 5) -> str:
    """
    Carica tutti i file di contesto dalla cartella context e restituisce un testo formattato.
    
    Args:
        context_dir: Percorso alla cartella context
        keywords: Keyword opzionali per filtrare sezioni rilevanti
        max_sections_per_file: Numero massimo di sezioni da includere per file
        
    Returns:
        str: Testo formattato con tutto il contesto caricato
    """
    context_path = Path(context_dir)
    
    if not context_path.exists():
        return ""
    
    context_parts = []
    keywords = keywords or []
    
    # Processa tutti i file nella cartella
    for file_path in sorted(context_path.iterdir()):
        if file_path.is_file():
            file_name = file_path.name
            
            # Salta file nascosti e README
            if file_name.startswith('.') or file_name.lower() == 'readme.md':
                continue
            
            try:
                if file_path.suffix.lower() == '.pdf':
                    if not PDF_AVAILABLE:
                        continue
                    text = extract_text_from_pdf(file_path)
                    file_type = "PDF"
                elif file_path.suffix.lower() in ['.md', '.markdown']:
                    text = extract_text_from_markdown(file_path)
                    file_type = "Markdown"
                else:
                    continue
                
                if not text or text.startswith("[ERRORE"):
                    continue
                
                # Dividi in sezioni
                sections = split_into_sections(text)
                
                # Se ci sono keyword, filtra sezioni rilevanti
                if keywords and sections:
                    relevant_sections = find_relevant_sections(
                        sections, keywords, max_sections_per_file
                    )
                else:
                    # Se non ci sono keyword o il file è piccolo, includi tutto
                    if len(sections) <= max_sections_per_file:
                        relevant_sections = sections
                    else:
                        # Prendi le prime sezioni
                        relevant_sections = sections[:max_sections_per_file]
                
                # Formatta il contenuto
                if relevant_sections:
                    context_parts.append(f"\n=== CONTESTO DA {file_type}: {file_name} ===\n")
                    for section in relevant_sections:
                        context_parts.append(f"\n## {section['title']}\n")
                        context_parts.append(section['content'])
                        context_parts.append("\n")
            
            except Exception as e:
                # Continua con altri file anche se uno fallisce
                logger.warning(f"Errore durante il caricamento del file {file_name}: {e}. Continuo con altri file.")
                continue
    
    return "\n".join(context_parts)


def get_context_for_analysis(analysis_type: str = "general",
                            field_name: Optional[str] = None) -> str:
    """
    Ottiene il contesto rilevante per un tipo specifico di analisi.
    
    Args:
        analysis_type: Tipo di analisi ('data_summary', 'error_patterns', 'chart_commentary', 'general')
        field_name: Nome del campo specifico (opzionale)
        
    Returns:
        str: Contesto formattato
    """
    # Keyword specifiche per tipo di analisi
    keyword_map = {
        'data_summary': ['dizionario', 'dati', 'campo', 'colonna', 'dataset', 'field', 'terminologia', 'sistema'],
        'error_patterns': ['errore', 'error', 'pattern', 'validazione', 'confronto', 'FP', 'FN', 'TP', 'TN', 'comparison'],
        'chart_commentary': ['metodo', 'performance', 'metriche', 'precision', 'recall', 'accuracy', 'f1', 'grafico'],
        'general': []
    }
    
    keywords = keyword_map.get(analysis_type, [])
    
    # Aggiungi keyword specifiche per field_name
    if field_name:
        keywords.append(field_name.lower())
        keywords.append('campo')
        keywords.append('field')
    
    return load_context_files(keywords=keywords if keywords else None)
