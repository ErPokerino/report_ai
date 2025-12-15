"""
Modulo per l'integrazione con LangChain per analisi AI.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import pandas as pd
import os
import sys
from typing import Dict, List, Optional, Any, Union
from openai import RateLimitError
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import logging

# Importa funzione helper per calcolo percentuali
try:
    from .data_loader import calculate_percentage
except ImportError:
    # Fallback per quando eseguito da notebook
    from data_loader import calculate_percentage

# Carica variabili d'ambiente da .env
load_dotenv()

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

# Timeout per le chiamate API (in secondi) - configurabile via variabile d'ambiente
API_TIMEOUT_SECONDS = int(os.getenv("LLM_API_TIMEOUT", "60"))

# Costanti
GEMINI_MODEL = "gemini-3-pro-preview"
GEMINI_FLASH_MODEL = "gemini-2.5-flash"
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_TEMPERATURE = 0

# Importa il modulo per caricare il contesto
# Prova prima import relativo, poi assoluto per compatibilità con notebook Quarto
CONTEXT_LOADER_AVAILABLE = False
try:
    # Prova import relativo (quando eseguito come modulo)
    from .context_loader import get_context_for_analysis
    CONTEXT_LOADER_AVAILABLE = True
except ImportError as e:
    try:
        # Fallback a import assoluto (quando eseguito da notebook)
        from context_loader import get_context_for_analysis
        CONTEXT_LOADER_AVAILABLE = True
    except ImportError as e2:
        # Se entrambi falliscono, usa funzione stub
        logger.warning(f"Impossibile importare context_loader: {e2}. Funzionalità di contesto disabilitata.")
        CONTEXT_LOADER_AVAILABLE = False
        def get_context_for_analysis(analysis_type: str = "general", field_name: Optional[str] = None) -> str:
            return ""

# Importa Gemini se disponibile
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError as e:
    logger.debug(f"langchain-google-genai non disponibile: {e}. Supporto Gemini disabilitato.")
    GEMINI_AVAILABLE = False

# Importa ModelTracker per tracking chiamate LLM
try:
    from .model_tracker import ModelTracker
except ImportError:
    # Fallback per quando eseguito da notebook
    try:
        from model_tracker import ModelTracker
    except ImportError:
        # Se non disponibile, crea stub
        logger.warning("ModelTracker non disponibile. Tracking modelli disabilitato.")
        class ModelTracker:
            def track_call(self, model_name: str, success: bool = True):
                pass
            def get_usage_stats(self):
                return {}
            def get_primary_model(self):
                return None

# Crea istanza globale del tracker
tracker = ModelTracker()


def get_model_display_name(model_name: Optional[str]) -> str:
    """
    Converte il nome tecnico del modello in un nome leggibile per il report.
    
    Args:
        model_name: Nome tecnico del modello (es. "gpt-5.2", "gemini-3-pro-preview")
        
    Returns:
        Nome formattato per la visualizzazione (es. "GPT-5.2", "Gemini 3 PRO")
    """
    if model_name is None:
        return "modelli LLM disponibili"
    
    model_map = {
        "gpt-5.2": "GPT-5.2",
        "gpt-4o": "GPT-4o",
        "gpt-4-turbo": "GPT-4 Turbo",
        "gpt-4": "GPT-4",
        "gemini-3-pro-preview": "Gemini 3 PRO",
        "gemini-3-pro": "Gemini 3 PRO",
        "gemini-2.5-flash": "Gemini Flash 2.5",
    }
    return model_map.get(model_name, model_name.upper().replace("-", " "))


def get_llm(model_name: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> Optional[Dict[str, Any]]:
    """
    Inizializza il modello LLM.
    
    Args:
        model_name: Nome del modello da usare
        temperature: Temperatura per la generazione
        
    Returns:
        Configurazione LLM o None se API key non disponibile
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() == "":
        return None
    
    # Non inizializzare ChatOpenAI qui - fallo solo quando necessario
    # Questo evita errori durante l'inizializzazione se la quota è esaurita
    return {"model_name": model_name, "temperature": temperature, "api_key": api_key}


def get_llm_with_fallback(primary_model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> Optional[Dict[str, Any]]:
    """
    Inizializza LLM con fallback a modelli alternativi se quota esaurita.
    
    Ordine di fallback: gpt-5.2 -> gemini-3-pro-preview -> gemini-2.5-flash -> gpt-4o
    
    Args:
        primary_model: Modello principale da provare (default: gpt-5.2)
        temperature: Temperatura per la generazione
        
    Returns:
        Configurazione LLM con model_name, temperature, api_key, provider, o None se tutti falliscono
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    # Se non ci sono API keys disponibili, restituisci None
    if (not openai_api_key or openai_api_key.strip() == "") and (not google_api_key or google_api_key.strip() == ""):
        return None
    
    # Ordine di fallback richiesto: gpt-5.2 -> gemini-3-pro-preview -> gemini-2.5-flash -> gpt-4o
    fallback_models = ["gpt-5.2"]
    
    # Aggiungi Gemini se disponibile
    if GEMINI_AVAILABLE and google_api_key and google_api_key.strip() != "":
        fallback_models.append(GEMINI_MODEL)
        fallback_models.append(GEMINI_FLASH_MODEL)
    
    # Aggiungi GPT-4o
    fallback_models.append("gpt-4o")
    
    # Se il modello primario non è nella lista, aggiungilo all'inizio
    if primary_model not in fallback_models:
        fallback_models.insert(0, primary_model)
    else:
        # Riordina per avere primary_model come primo
        if primary_model in fallback_models:
            fallback_models.remove(primary_model)
        fallback_models.insert(0, primary_model)
    
    # Prova ogni modello nell'ordine specificato
    for model in fallback_models:
        # Determina se è un modello OpenAI o Gemini
        is_gemini = model == GEMINI_MODEL or model == GEMINI_FLASH_MODEL
        
        # Prova OpenAI
        if not is_gemini and openai_api_key and openai_api_key.strip() != "":
            try:
                llm = ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    api_key=openai_api_key,
                    timeout=API_TIMEOUT_SECONDS
                )
                return {
                    "model_name": model, 
                    "temperature": temperature, 
                    "api_key": openai_api_key, 
                    "llm": llm, 
                    "fallback_models": fallback_models,
                    "provider": "openai"
                }
            except RateLimitError as e:
                logger.debug(f"RateLimitError per modello {model}: {e}. Provo modello successivo.")
                continue
            except Exception as e:
                logger.debug(f"Errore inizializzazione modello {model}: {e}. Provo modello successivo.")
                continue
        
        # Prova Gemini
        elif is_gemini and GEMINI_AVAILABLE and google_api_key and google_api_key.strip() != "":
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    google_api_key=google_api_key
                )
                return {
                    "model_name": model, 
                    "temperature": temperature, 
                    "api_key": google_api_key, 
                    "llm": llm, 
                    "fallback_models": fallback_models,
                    "provider": "google"
                }
            except Exception as e:
                logger.debug(f"Errore inizializzazione {model}: {e}. Provo modello successivo.")
                continue
    
    # Se tutti i modelli falliscono, restituisci None
    return None


def _extract_text_from_response(response: Any) -> str:
    """
    Estrae il testo da una risposta LLM, gestendo sia OpenAI (.content) che Gemini (.text).
    
    Secondo la documentazione LangChain:
    - Gemini: response.text dovrebbe essere una stringa consolidata, response.content può essere lista di dizionari
    - OpenAI: response.content è una stringa
    
    Args:
        response: Risposta da LLM (OpenAI o Gemini) - oggetto AIMessage
        
    Returns:
        Testo estratto dalla risposta
    """
    # Se la risposta stessa è una lista (caso anomalo)
    if isinstance(response, list):
        extracted = []
        for item in response:
            if isinstance(item, dict):
                # Formato: {"type": "text", "text": "..."}
                if 'text' in item:
                    extracted.append(item.get('text', ''))
            else:
                extracted.append(str(item))
        return ' '.join(extracted)
    
    # Priorità 1: Per Gemini, response.text dovrebbe già essere una stringa consolidata
    if hasattr(response, 'text'):
        text = response.text
        # Se text è già una stringa (caso normale Gemini), restituiscila
        if isinstance(text, str):
            return text
        # Se text è una lista (caso anomalo), gestiscila
        if isinstance(text, list):
            extracted = []
            for item in text:
                if isinstance(item, dict) and 'text' in item:
                    extracted.append(item.get('text', ''))
                else:
                    extracted.append(str(item))
            return ' '.join(extracted)
    
    # Priorità 2: Fallback - usa response.content
    content = getattr(response, 'content', None)
    
    # Se content è una lista di dizionari (formato Gemini: [{"type": "text", "text": "..."}])
    if isinstance(content, list):
        extracted = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                # Estrai il campo "text" dal dizionario
                extracted.append(item.get('text', ''))
            else:
                extracted.append(str(item))
        return ' '.join(extracted)
    
    # Se content è una stringa (OpenAI o Gemini caso normale)
    if isinstance(content, str):
        return content
    
    # Se content è None o altro tipo, prova altri attributi
    if content is None:
        # Prova a vedere se response ha altri attributi
        if hasattr(response, 'parts'):
            parts = response.parts
            if isinstance(parts, list):
                extracted = []
                for item in parts:
                    if isinstance(item, dict) and 'text' in item:
                        extracted.append(item.get('text', ''))
                    else:
                        extracted.append(str(item))
                return ' '.join(extracted)
            return str(parts) if parts is not None else ""
    
    # Ultimo fallback: converti in stringa
    return str(content) if content is not None else ""


def _invoke_with_timeout(llm: Any, messages: List[HumanMessage], timeout_seconds: int = API_TIMEOUT_SECONDS) -> Any:
    """
    Invoca LLM con timeout per evitare blocchi indefiniti.
    
    Args:
        llm: Istanza LLM da invocare
        messages: Lista di messaggi da inviare
        timeout_seconds: Timeout in secondi (default: API_TIMEOUT_SECONDS)
        
    Returns:
        Risposta LLM
        
    Raises:
        FuturesTimeoutError: Se la chiamata supera il timeout
        Exception: Altri errori dalla chiamata LLM
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(llm.invoke, messages)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            logger.warning(f"Timeout dopo {timeout_seconds} secondi per la chiamata LLM")
            raise


def _try_fallback_models(
    fallback_models: List[str],
    current_index: int,
    prompt: str,
    temperature: float,
    openai_api_key: Optional[str],
    google_api_key: Optional[str]
) -> Optional[str]:
    """
    Prova modelli fallback in sequenza.
    
    Args:
        fallback_models: Lista di modelli da provare
        current_index: Indice del modello corrente nella lista
        prompt: Prompt da inviare
        temperature: Temperatura per la generazione
        openai_api_key: API key OpenAI
        google_api_key: API key Google
        
    Returns:
        Risposta LLM o None se tutti i modelli falliscono
    """
    # Prova modelli successivi nell'ordine specificato
    for model in fallback_models[current_index + 1:]:
        # Determina se è un modello OpenAI o Gemini
        is_gemini = model == GEMINI_MODEL or model == GEMINI_FLASH_MODEL
        
        # Prova OpenAI
        if not is_gemini and openai_api_key and openai_api_key.strip() != "":
            try:
                logger.debug(f"Tentativo fallback con modello OpenAI: {model}")
                llm = ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    api_key=openai_api_key,
                    timeout=API_TIMEOUT_SECONDS
                )
                response = _invoke_with_timeout(llm, [HumanMessage(content=prompt)])
                result = _extract_text_from_response(response)
                logger.info(f"Fallback riuscito con modello: {model}")
                # Traccia fallback riuscito
                tracker.track_call(model, success=True)
                return result
            except (RateLimitError, FuturesTimeoutError) as e:
                logger.debug(f"Errore con modello fallback {model}: {e}. Provo successivo.")
                continue
            except Exception as e:
                logger.debug(f"Errore generico con modello fallback {model}: {e}. Provo successivo.")
                continue
        
        # Prova Gemini
        elif is_gemini and GEMINI_AVAILABLE and google_api_key and google_api_key.strip() != "":
            try:
                logger.debug(f"Tentativo fallback con Gemini: {model}")
                llm = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    google_api_key=google_api_key
                )
                response = _invoke_with_timeout(llm, [HumanMessage(content=prompt)])
                result = _extract_text_from_response(response)
                logger.info(f"Fallback riuscito con Gemini: {model}")
                # Traccia fallback Gemini riuscito
                tracker.track_call(model, success=True)
                return result
            except Exception as e:
                logger.debug(f"Errore con Gemini fallback {model}: {e}. Provo successivo.")
                continue
    
    return None


def invoke_llm_with_fallback(llm_config: Optional[Dict[str, Any]], prompt: str) -> Optional[str]:
    """
    Invoca LLM con fallback automatico se si verifica RateLimitError o timeout.
    
    Args:
        llm_config: Configurazione LLM da get_llm_with_fallback()
        prompt: Prompt da inviare all'LLM
        
    Returns:
        Risposta dell'LLM, o None se tutti i modelli falliscono
    """
    if llm_config is None:
        return None
    
    # Ottieni llm e lista di fallback
    llm = llm_config.get("llm")
    fallback_models = llm_config.get("fallback_models", [llm_config["model_name"]])
    provider = llm_config.get("provider", "openai")
    model_name = llm_config.get("model_name")
    
    # Trova l'indice del modello corrente in modo sicuro
    if model_name in fallback_models:
        current_model_index = fallback_models.index(model_name)
    else:
        # Se il modello non è nella lista, aggiungilo all'inizio e usa indice 0
        fallback_models.insert(0, model_name)
        current_model_index = 0
    
    temperature = llm_config.get("temperature", DEFAULT_TEMPERATURE)
    
    # Prova prima con il modello corrente
    try:
        logger.debug(f"Invio richiesta LLM con modello: {model_name}")
        response = _invoke_with_timeout(llm, [HumanMessage(content=prompt)])
        result = _extract_text_from_response(response)
        # Traccia chiamata riuscita
        tracker.track_call(model_name, success=True)
        return result
    except (RateLimitError, FuturesTimeoutError) as e:
        # Se quota esaurita o timeout, prova fallback
        logger.warning(f"Errore con modello {model_name}: {e}. Tentativo fallback.")
        return _try_fallback_models(
            fallback_models,
            current_model_index,
            prompt,
            temperature,
            os.getenv("OPENAI_API_KEY"),
            os.getenv("GOOGLE_API_KEY")
        )
    except Exception as e:
        # Altri errori: se è OpenAI, prova fallback; se è già Gemini, restituisci None
        logger.warning(f"Errore generico con modello {model_name}: {e}")
        if provider == "openai":
            logger.info("Tentativo fallback per errore OpenAI")
            return _try_fallback_models(
                fallback_models,
                current_model_index,
                prompt,
                temperature,
                os.getenv("OPENAI_API_KEY"),
                os.getenv("GOOGLE_API_KEY")
            )
        logger.error(f"Errore con Gemini, nessun fallback disponibile")
        return None


def analyze_data_summary(df: pd.DataFrame, field_name: Optional[str] = None) -> str:
    """
    Genera un riassunto analitico dei dati usando AI.
    
    Args:
        df: DataFrame da analizzare
        field_name: Nome del campo specifico da analizzare (opzionale)
        
    Returns:
        Riassunto generato dall'AI, o messaggio di errore se AI non disponibile
    """
    llm_config = get_llm_with_fallback()
    if llm_config is None:
        return "Analisi AI non disponibile (API key non configurata o tutti i modelli hanno quota esaurita)."
    
    try:
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
                pct_validated = calculate_percentage(field_validated, field_count, decimal_places=1)
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

Dati validati: {validated_count} su {total_count} ({calculate_percentage(validated_count, total_count, decimal_places=1):.1f}%)

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
        
        # Usa la funzione con fallback automatico
        result = invoke_llm_with_fallback(llm_config, prompt)
        if result is not None:
            return result
        else:
            return "Analisi AI non disponibile (quota esaurita per tutti i modelli disponibili)."
    except Exception as e:
        # Se c'è un errore durante l'inizializzazione, restituisci un messaggio di fallback
        return f"Analisi AI non disponibile (errore inizializzazione: {str(e)[:100]})."


def generate_chart_commentary(
    chart_description: str,
    chart_data_summary: str,
    domain: str = "general",
    field_name: Optional[str] = None
) -> Optional[str]:
    """
    Genera un commento testuale per un grafico usando AI.
    
    Args:
        chart_description: Descrizione del tipo di grafico
        chart_data_summary: Riassunto dei dati visualizzati
        domain: Dominio dei dati ("document" per documentale, "general" per generale)
        field_name: Nome del campo specifico analizzato (opzionale)
        
    Returns:
        Commento generato dall'AI in formato markdown, o None se AI non disponibile
    """
    llm_config = get_llm_with_fallback()
    if llm_config is None:
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
    
    # Usa la funzione con fallback automatico
    return invoke_llm_with_fallback(llm_config, prompt)


def analyze_error_patterns(df: pd.DataFrame, field_name: Optional[str] = None) -> Optional[str]:
    """
    Analizza i pattern di errore nei dati Lucy.
    
    Args:
        df: DataFrame con dati validati
        field_name: Nome del campo specifico da analizzare (opzionale)
        
    Returns:
        Analisi dei pattern di errore, o None se AI non disponibile
    """
    llm_config = get_llm_with_fallback()
    if llm_config is None:
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
    
    # Usa la funzione con fallback automatico
    return invoke_llm_with_fallback(llm_config, prompt)


def generate_section_text(section_topic: str, data_context: str) -> Optional[str]:
    """
    Genera testo per una sezione del report usando AI.
    
    Args:
        section_topic: Argomento della sezione
        data_context: Contesto dei dati
        
    Returns:
        Testo generato, o None se AI non disponibile
    """
    llm_config = get_llm_with_fallback()
    if llm_config is None:
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
    
    # Usa la funzione con fallback automatico
    return invoke_llm_with_fallback(llm_config, prompt)
