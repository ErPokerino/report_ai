"""
Modulo per l'integrazione con LangChain per analisi AI.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import pandas as pd
import os
from openai import RateLimitError

# Carica variabili d'ambiente da .env
load_dotenv()

# Importa il modulo per caricare il contesto
try:
    from context_loader import get_context_for_analysis
    CONTEXT_LOADER_AVAILABLE = True
except ImportError:
    CONTEXT_LOADER_AVAILABLE = False
    def get_context_for_analysis(analysis_type="general", field_name=None):
        return ""


def get_llm(model_name="gpt-5.2", temperature=0):
    """
    Inizializza il modello LLM.
    
    Args:
        model_name: Nome del modello da usare
        temperature: Temperatura per la generazione
        
    Returns:
        ChatOpenAI: Istanza del modello, o None se API key non disponibile o errore
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() == "":
        return None
    
    # Non inizializzare ChatOpenAI qui - fallo solo quando necessario
    # Questo evita errori durante l'inizializzazione se la quota è esaurita
    return {"model_name": model_name, "temperature": temperature, "api_key": api_key}


def analyze_data_summary(df, field_name=None):
    """
    Genera un riassunto analitico dei dati usando AI.
    
    Args:
        df: DataFrame da analizzare
        field_name: Nome del campo specifico da analizzare (opzionale)
        
    Returns:
        str: Riassunto generato dall'AI, o None se AI non disponibile
    """
    llm_config = get_llm()
    if llm_config is None:
        return "Analisi AI non disponibile (API key non configurata)."
    
    try:
        llm = ChatOpenAI(
            model=llm_config["model_name"], 
            temperature=llm_config["temperature"], 
            api_key=llm_config["api_key"]
        )
        # Prepara statistiche descrittive
        stats = df.describe().to_string()
        shape_info = f"Shape: {df.shape}\nColonne: {', '.join(df.columns.tolist())}"
        
        # Controlla se sono dati Lucy
        is_lucy_data = 'datetime_sent' in df.columns or 'method_pred' in df.columns
        
        if is_lucy_data:
            # Analisi specifica per dati Lucy
            validated_count = df['is_validated'].sum() if 'is_validated' in df.columns else 0
            total_count = len(df)
            methods = df['method_pred'].value_counts().to_string() if 'method_pred' in df.columns else "N/A"
            
            # Aggiungi informazioni su field_name se specificato
            field_context = ""
            if field_name:
                field_context = f"\n\nAnalisi specifica per il campo: **{field_name}**\n"
                field_count = len(df[df['field_name'] == field_name]) if 'field_name' in df.columns else 0
                field_validated = len(df[(df['field_name'] == field_name) & (df['is_validated'])]) if 'field_name' in df.columns else 0
                field_context += f"Record totali per questo campo: {field_count}\n"
                pct_validated = (field_validated/field_count*100) if field_count > 0 else 0
                field_context += f"Record validati per questo campo: {field_validated} ({pct_validated:.1f}%)\n"
            elif 'field_name' in df.columns:
                field_names = df['field_name'].value_counts().to_string()
                field_context = f"\n\nDistribuzione campi (field_name):\n{field_names}\n"
            
            # Aggiorna il contesto per menzionare tutti i campi, non solo id_subject
            field_description = f"tutti i campi indicati in field_name" if 'field_name' in df.columns else "informazioni dalle fatture"
            
            # Carica contesto rilevante dalla cartella context
            context_text = ""
            if CONTEXT_LOADER_AVAILABLE:
                context_text = get_context_for_analysis('data_summary', field_name)
                if context_text:
                    context_text = f"\n\n=== CONTESTO DI DOMINIO E DOCUMENTAZIONE ===\n{context_text}\n"
            
            prompt = f"""Analizza i seguenti dati di riconoscimento documentale (fatture) e fornisci un riassunto analitico conciso usando markdown.

Contesto: Dati di un sistema di riconoscimento automatico di informazioni da fatture. Il sistema estrae {field_description}. I dati includono predizioni di vari algoritmi e validazioni umane.{field_context}{context_text}

{shape_info}

Statistiche descrittive:
{stats}

Dati validati: {validated_count} su {total_count} ({validated_count/total_count*100:.1f}%)

Distribuzione metodi:
{methods}

Fornisci un'analisi strutturata che evidenzi:
1. Volume e copertura dei dati (quanti record validati vs totali)
2. Performance generale dei metodi di riconoscimento
3. Osservazioni rilevanti sul sistema di validazione

Formattazione richiesta:
- Usa *corsivo* per nomi di metodi (es. *azure_model*, *query-vat_number*), termini tecnici (es. *fallback*, *ensemble*), e nomi di sistemi
- Usa elenchi puntati quando appropriato per organizzare informazioni
- Mantieni paragrafi brevi e leggibili
- Evita muri di testo: struttura il contenuto in modo chiaro

IMPORTANTE: Usa markdown per la formattazione (elenchi, corsivo, grassetto quando necessario).
"""
        else:
            prompt = f"""Analizza i seguenti dati e fornisci un riassunto analitico conciso usando markdown.
        
{shape_info}

Statistiche descrittive:
{stats}

Fornisci un'analisi strutturata che evidenzi:
1. Caratteristiche principali dei dati
2. Pattern o tendenze evidenti
3. Osservazioni rilevanti

Formattazione richiesta:
- Usa *corsivo* per termini tecnici e nomi di sistemi quando appropriato
- Usa elenchi puntati quando appropriato per organizzare informazioni
- Mantieni paragrafi brevi e leggibili
- Evita muri di testo: struttura il contenuto in modo chiaro

IMPORTANTE: Usa markdown per la formattazione (elenchi, corsivo, grassetto quando necessario).
"""
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except (RateLimitError, Exception) as e:
            # Se c'è un errore (es. quota esaurita), restituisci un messaggio di fallback
            return f"Analisi AI non disponibile (errore: {str(e)[:100]})."
    except (RateLimitError, Exception) as e:
        # Se c'è un errore durante l'inizializzazione, restituisci un messaggio di fallback
        return f"Analisi AI non disponibile (errore inizializzazione: {str(e)[:100]})."


def generate_chart_commentary(chart_description, chart_data_summary, domain="general", field_name=None):
    """
    Genera un commento testuale per un grafico usando AI.
    
    Args:
        chart_description: Descrizione del tipo di grafico
        chart_data_summary: Riassunto dei dati visualizzati
        domain: Dominio dei dati ("document" per documentale, "general" per generale)
        field_name: Nome del campo specifico analizzato (opzionale)
        
    Returns:
        str: Commento generato dall'AI in formato markdown
    """
    llm_config = get_llm()
    if llm_config is None:
        return None
    
    try:
        llm = ChatOpenAI(
            model=llm_config["model_name"], 
            temperature=llm_config["temperature"], 
            api_key=llm_config["api_key"]
        )
    except (RateLimitError, Exception) as e:
        # Gestisci errori di quota o altri errori
        return None
    
    domain_context = ""
    if domain == "document":
        domain_context = "Contesto: Analisi di performance di algoritmi di riconoscimento documentale per fatture. "
    
    field_context = ""
    if field_name:
        field_context = f"\n\nNota: Questo grafico mostra dati specifici per il campo **{field_name}**. "
    
    # Carica contesto rilevante dalla cartella context
    context_text = ""
    if CONTEXT_LOADER_AVAILABLE:
        context_text = get_context_for_analysis('chart_commentary', field_name)
        if context_text:
            context_text = f"\n\n=== CONTESTO DI DOMINIO E DOCUMENTAZIONE ===\n{context_text}\n"
    
    prompt = f"""Genera un commento analitico professionale per il seguente grafico usando markdown.

{domain_context}{field_context}{context_text}Tipo di grafico: {chart_description}

Dati visualizzati:
{chart_data_summary}

Il commento deve:
1. Iniziare con 1-2 paragrafi che descrivono i pattern principali osservati
2. Includere una sezione "Punti di interesse/anomalie" con elenco puntato quando applicabile
3. Includere una sezione "Raccomandazioni operative" con elenco puntato quando applicabile

Formattazione richiesta:
- Usa *corsivo* per nomi di metodi (es. *azure_model*, *query-vat_number*), termini tecnici (es. *fallback*, *ensemble*, *routing*, *A/B test*), e nomi di sistemi/funzionalità
- Usa elenchi puntati per organizzare informazioni multiple (punti chiave, anomalie, raccomandazioni)
- Mantieni paragrafi brevi e leggibili
- Evita muri di testo: struttura il contenuto in sezioni chiare
- Usa grassetto (**testo**) per enfatizzare concetti importanti quando necessario

IMPORTANTE: Usa markdown per la formattazione (elenchi, corsivo, grassetto quando necessario). Struttura il commento in modo leggibile con paragrafi brevi e elenchi quando appropriato.
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except (RateLimitError, Exception) as e:
        # Se c'è un errore (es. quota esaurita), restituisci un messaggio di fallback
        return f"Analisi AI non disponibile (errore: {str(e)[:100]})."


def analyze_error_patterns(df, field_name=None):
    """
    Analizza i pattern di errore nei dati Lucy.
    
    Args:
        df: DataFrame con dati validati
        field_name: Nome del campo specifico da analizzare (opzionale)
        
    Returns:
        str: Analisi dei pattern di errore
    """
    llm_config = get_llm()
    if llm_config is None:
        return None
    
    try:
        llm = ChatOpenAI(
            model=llm_config["model_name"], 
            temperature=llm_config["temperature"], 
            api_key=llm_config["api_key"]
        )
    except (RateLimitError, Exception) as e:
        # Gestisci errori di quota o altri errori
        return None
    
    # Filtra per field_name se specificato
    if field_name and 'field_name' in df.columns:
        df = df[df['field_name'] == field_name].copy()
    
    validated = df[df['is_validated']].copy() if 'is_validated' in df.columns else df
    
    if len(validated) == 0:
        field_msg = f" per il campo {field_name}" if field_name else ""
        return f"Nessun dato validato disponibile{field_msg} per l'analisi degli errori."
    
    # Analizza errori
    fp_data = validated[validated['comparison'] == 'FP']
    fn_data = validated[validated['comparison'] == 'FN']
    
    fp_by_method = fp_data['method_pred'].value_counts().to_string() if len(fp_data) > 0 else "Nessun FP"
    fn_by_method = fn_data['method_pred'].value_counts().to_string() if len(fn_data) > 0 else "Nessun FN"
    
    fp_avg_confidence = fp_data['confidence'].mean() if 'confidence' in fp_data.columns and len(fp_data) > 0 else None
    fn_avg_confidence = fn_data['confidence'].mean() if 'confidence' in fn_data.columns and len(fn_data) > 0 else None
    
    # Formatta confidence per la stringa
    fp_conf_str = f"{fp_avg_confidence:.3f}" if fp_avg_confidence is not None else 'N/A'
    fn_conf_str = f"{fn_avg_confidence:.3f}" if fn_avg_confidence is not None else 'N/A'
    
    field_context = ""
    if field_name:
        field_context = f"\n\nAnalisi specifica per il campo: **{field_name}**\n"
        field_total = len(validated)
        field_context += f"Record validati per questo campo: {field_total}\n"
    
    # Carica contesto rilevante dalla cartella context
    context_text = ""
    if CONTEXT_LOADER_AVAILABLE:
        context_text = get_context_for_analysis('error_patterns', field_name)
        if context_text:
            context_text = f"\n\n=== CONTESTO DI DOMINIO E DOCUMENTAZIONE ===\n{context_text}\n"
    
    prompt = f"""Analizza i pattern di errore in un sistema di riconoscimento documentale e fornisci un'analisi concisa usando markdown.

Contesto: Sistema di riconoscimento automatico di informazioni da fatture. False Positive (FP) = predetto positivo ma reale negativo. False Negative (FN) = predetto negativo ma reale positivo.{field_context}{context_text}

False Positive per metodo:
{fp_by_method}

False Negative per metodo:
{fn_by_method}

Confidence media FP: {fp_conf_str}
Confidence media FN: {fn_conf_str}

Fornisci un'analisi strutturata che:
1. Identifichi quali metodi hanno più problemi (FP o FN)
2. Analizzi se la confidence è correlata agli errori
3. Suggerisca possibili miglioramenti o aree di attenzione

Formattazione richiesta:
- Usa *corsivo* per nomi di metodi (es. *azure_model*, *query-vat_number*), termini tecnici (es. *fallback*, *ensemble*, *routing*), e nomi di sistemi
- Usa elenchi puntati per organizzare punti chiave, anomalie e raccomandazioni
- Mantieni paragrafi brevi e leggibili
- Evita muri di testo: struttura il contenuto in sezioni chiare
- Usa grassetto (**testo**) per enfatizzare concetti importanti quando necessario

IMPORTANTE: Usa markdown per la formattazione (elenchi, corsivo, grassetto quando necessario). Struttura il commento in modo leggibile con paragrafi brevi e elenchi quando appropriato.
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except (RateLimitError, Exception) as e:
        # Se c'è un errore (es. quota esaurita), restituisci un messaggio di fallback
        return f"Analisi AI non disponibile (errore: {str(e)[:100]})."


def generate_section_text(section_topic, data_context):
    """
    Genera testo per una sezione del report usando AI.
    
    Args:
        section_topic: Argomento della sezione
        data_context: Contesto dei dati
        
    Returns:
        str: Testo generato
    """
    llm_config = get_llm()
    if llm_config is None:
        return None
    
    try:
        llm = ChatOpenAI(
            model=llm_config["model_name"], 
            temperature=llm_config["temperature"], 
            api_key=llm_config["api_key"]
        )
    except (RateLimitError, Exception) as e:
        # Gestisci errori di quota o altri errori
        return None
    
    # Carica contesto rilevante dalla cartella context
    context_text = ""
    if CONTEXT_LOADER_AVAILABLE:
        context_text = get_context_for_analysis('general')
        if context_text:
            context_text = f"\n\n=== CONTESTO DI DOMINIO E DOCUMENTAZIONE ===\n{context_text}\n"
    
    prompt = f"""Scrivi una sezione di report professionale (2-3 paragrafi) su:

Argomento: {section_topic}

Contesto dati:
{data_context}{context_text}

Il testo dovrebbe essere:
- Professionale e chiaro
- Basato sui dati forniti
- Strutturato logicamente

IMPORTANTE: Scrivi solo testo normale, senza asterischi, cancelletto o altri simboli di formattazione markdown.
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except (RateLimitError, Exception) as e:
        # Se c'è un errore (es. quota esaurita), restituisci un messaggio di fallback
        return f"Analisi AI non disponibile (errore: {str(e)[:100]})."

