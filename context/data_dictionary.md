# Dizionario Dati - Dataset Lucy

## Panoramica

Questo documento descrive le colonne del dataset `lucy_data.csv` utilizzato per l'analisi del sistema Lucy.

## Colonne del Dataset

### `protocol`
- **Tipo**: Numerico (intero)
- **Descrizione**: Numero di protocollo univoco della fattura
- **Esempio**: `29077777`
- **Note**: Identificatore univoco per ogni fattura processata

### `id_company`
- **Tipo**: Numerico (intero)
- **Descrizione**: Identificativo dell'azienda Luxottica che ha ricevuto la fattura
- **Esempio**: `254`, `59`
- **Note**: Può variare da 1 a 264 (circa 213 valori unici nel dataset)

### `country`
- **Tipo**: Stringa (codice paese ISO)
- **Descrizione**: Codice paese di origine della fattura
- **Esempio**: `DE`, `GB`, `IT`, `FR`
- **Note**: Utilizzato per analisi geografiche e routing dei metodi

### `field_name`
- **Tipo**: Stringa
- **Descrizione**: Nome del campo estratto dalla fattura
- **Note**: Campo target del sistema di riconoscimento

### `prediction`
- **Tipo**: Numerico (intero) o NaN
- **Descrizione**: Valore predetto dal sistema per il campo `id_subject` (codice fornitore)
- **Esempio**: `573444`, `571981`
- **Note**: Può essere NaN se il sistema non ha prodotto una predizione

### `truth`
- **Tipo**: Numerico (intero) o NaN
- **Descrizione**: Valore reale/validato manualmente per il campo `id_subject`
- **Esempio**: `571981`, `574070`
- **Note**: Disponibile solo per record validati (circa 23-25% del totale)

### `comparison`
- **Tipo**: Stringa categorica o NaN
- **Valori possibili**:
  - `TP` (True Positive): Predizione corretta (predetto positivo, reale positivo)
  - `FP` (False Positive): Predizione errata (predetto positivo, reale negativo)
  - `FN` (False Negative): Predizione mancante (predetto negativo, reale positivo)
  - `TN` (True Negative): Predizione corretta (predetto negativo, reale negativo)
- **Note**: Disponibile solo per record validati

### `method_pred`
- **Tipo**: Stringa categorica
- **Descrizione**: Metodo utilizzato dal sistema per produrre la 

### `confidence`
- **Tipo**: Numerico (float, 0-1) o NaN
- **Descrizione**: Livello di confidenza della predizione
- **Range**: 0.0 - 1.0
- **Esempio**: `0.998`, `0.985`, `0.91`
- **Note**: Indica quanto il sistema è "sicuro" della predizione

### `datetime_sent`
- **Tipo**: Timestamp (datetime)
- **Descrizione**: Data e ora in cui la predizione è stata generata/inviata
- **Formato**: `YYYY-MM-DD HH:MM:SS.ffffff UTC`
- **Esempio**: `2025-12-08 12:24:10.000000 UTC`
- **Note**: Utilizzato per analisi temporali e timeline

### `autovalidated`
- **Tipo**: Booleano
- **Descrizione**: Flag che indica se il sistema ha autovalidato il documento, ritenendo che i campi principali riconosciuti siano corretti
- **Valori**: `True`, `False`, o NaN
- **Note**: 
  - Valutato da un modello ML separato che prende in input: campi predetti, metodo utilizzato e confidence
  - Se `autovalidated = True`, il documento salta il passaggio di validazione umana
  - Campo rilevante per l'analisi del flusso di validazione e dell'efficienza del sistema
  - Può essere utilizzato per analizzare la capacità del sistema di auto-validare correttamente

## Campi Derivati (calcolati in `data_loader.py`)

### `date`
- **Tipo**: Date
- **Descrizione**: Data estratta da `datetime_sent`
- **Calcolato**: `datetime_sent.dt.date`

### `hour`
- **Tipo**: Numerico (0-23)
- **Descrizione**: Ora del giorno estratta da `datetime_sent`
- **Calcolato**: `datetime_sent.dt.hour`

### `day_of_week`
- **Tipo**: Stringa
- **Descrizione**: Giorno della settimana
- **Valori**: `Monday`, `Tuesday`, `Wednesday`, etc.
- **Calcolato**: `datetime_sent.dt.day_name()`

### `is_validated`
- **Tipo**: Booleano
- **Descrizione**: Flag che indica se il record è stato validato manualmente
- **Calcolato**: `comparison.notna()`

### `is_correct`
- **Tipo**: Booleano o NaN
- **Descrizione**: Flag che indica se la predizione è corretta
- **Calcolato**: `comparison == 'TP'` (solo per record validati)

### `method_type`
- **Tipo**: Stringa categorica
- **Descrizione**: Categoria del metodo utilizzato
- **Valori**: `ML`, `Query`, `Other`
- **Calcolato**: Categorizzazione di `method_pred`

## Relazioni tra Campi

- `is_validated = True` → `comparison` non è NaN
- `is_validated = True` → `truth` non è NaN (nella maggior parte dei casi)
- `comparison = 'TP'` → `prediction == truth`
- `method_type` deriva da `method_pred` secondo regole di categorizzazione
- `autovalidated = True` → Il documento ha saltato la validazione umana (se il sistema lo supporta)

## Note sul Dataset

**IMPORTANTE**: Il dataset è dinamico e può cambiare nel tempo. Le statistiche e le colonne possono variare tra versioni diverse del dataset.

### Colonne Variabili

Il dataset può includere colonne aggiuntive o modificate in base alle esigenze del sistema:
- Nuovi campi predetti (oltre a `id_subject`)
- Flag di validazione automatica (es. `autovalidated`)
- Metadati aggiuntivi sul processo di riconoscimento
- Informazioni sul flusso di validazione


## Note Importanti

- **Dataset dinamico**: Il dataset può cambiare nel tempo. Verificare sempre le colonne disponibili prima di utilizzare il dataset.
- I record non validati hanno `truth = NaN` e `comparison = NaN`
- La confidence può essere NaN per alcuni metodi
- Il campo `prediction` può essere NaN se nessun metodo ha prodotto una predizione
- Le metriche di performance vengono calcolate solo sui record validati
- **Autovalidazione**: Il campo `autovalidated` è rilevante per analizzare l'efficienza del sistema e la capacità di ridurre il carico di validazione umana
