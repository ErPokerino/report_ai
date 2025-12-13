"""
Modulo per il caricamento e la preparazione dei dati.
"""
import pandas as pd
import numpy as np


def load_sample_data():
    """
    Carica un dataset di esempio per dimostrazione con dati più realistici.
    
    Returns:
        pd.DataFrame: Dataset di esempio con dati sintetici
    """
    np.random.seed(42)
    n_samples = 180  # 6 mesi di dati
    
    # Crea date
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')
    
    # Trend crescente con stagionalità settimanale
    trend = np.linspace(1000, 1500, n_samples)
    weekly_seasonality = 100 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
    noise = np.random.normal(0, 80, n_samples)
    sales = trend + weekly_seasonality + noise
    sales = np.maximum(sales, 200)  # Minimo 200
    
    # Clienti correlati alle vendite ma con variabilità
    customers_base = (sales / 20).astype(int)
    customers = customers_base + np.random.poisson(10, n_samples)
    
    # Regioni con distribuzione non uniforme
    region_weights = [0.4, 0.35, 0.25]  # Nord più popoloso
    regions = np.random.choice(['Nord', 'Centro', 'Sud'], n_samples, p=region_weights)
    
    # Categorie prodotto con preferenze per regione
    categories = []
    for region in regions:
        if region == 'Nord':
            cat_probs = [0.5, 0.3, 0.2]  # Categoria A preferita
        elif region == 'Centro':
            cat_probs = [0.3, 0.4, 0.3]  # Categoria B preferita
        else:
            cat_probs = [0.2, 0.3, 0.5]  # Categoria C preferita
        categories.append(np.random.choice(['A', 'B', 'C'], p=cat_probs))
    
    # Aggiungi metriche aggiuntive
    revenue = sales * (1 + np.random.normal(0, 0.1, n_samples))
    profit_margin = np.random.uniform(0.15, 0.35, n_samples)
    profit = revenue * profit_margin
    
    # Giorno della settimana
    day_of_week = dates.day_name()
    
    data = {
        'date': dates,
        'day_of_week': day_of_week,
        'sales': sales.round(2),
        'revenue': revenue.round(2),
        'profit': profit.round(2),
        'profit_margin': profit_margin.round(3),
        'customers': customers,
        'region': regions,
        'product_category': categories
    }
    
    df = pd.DataFrame(data)
    return df


def load_csv_data(filepath):
    """
    Carica dati da un file CSV.
    
    Args:
        filepath: Percorso al file CSV
        
    Returns:
        pd.DataFrame: Dati caricati
    """
    df = pd.read_csv(filepath)
    
    # Se è il dataset Lucy, prepara i dati
    if 'lucy_data' in filepath.lower() or 'datetime_sent' in df.columns:
        df = prepare_lucy_data(df)
    
    return df


def prepare_lucy_data(df):
    """
    Prepara i dati Lucy per l'analisi.
    
    Args:
        df: DataFrame con dati Lucy
        
    Returns:
        pd.DataFrame: DataFrame preparato con metriche calcolate
    """
    df = df.copy()
    
    # Converti datetime
    df['datetime_sent'] = pd.to_datetime(df['datetime_sent'])
    df['date'] = df['datetime_sent'].dt.date
    df['hour'] = df['datetime_sent'].dt.hour
    df['day_of_week'] = df['datetime_sent'].dt.day_name()
    
    # Crea flag per record validati
    df['is_validated'] = df['comparison'].notna()
    
    # Calcola accuratezza per record validati
    df['is_correct'] = df['comparison'].apply(lambda x: x == 'TP' if pd.notna(x) else None)
    
    # Categorizza metodi (ML vs Query)
    df['method_type'] = df['method_pred'].apply(categorize_method)
    
    return df


def categorize_method(method):
    """
    Categorizza il metodo di riconoscimento.
    
    Args:
        method: Nome del metodo
        
    Returns:
        str: 'ML' per machine learning, 'Query' per query-based, 'Other' per altri
    """
    if pd.isna(method):
        return 'Unknown'
    
    method_str = str(method).lower()
    
    if 'azure' in method_str or 'model' in method_str or 'ml' in method_str:
        return 'ML'
    elif 'query' in method_str:
        return 'Query'
    elif 'similarity' in method_str or 'logo' in method_str:
        return 'Other'
    else:
        return 'Other'


def calculate_metrics_by_method(df):
    """
    Calcola metriche di performance per ogni metodo.
    
    Args:
        df: DataFrame con dati validati
        
    Returns:
        pd.DataFrame: Metriche per metodo (precision, recall, F1, accuracy)
    """
    validated = df[df['is_validated']].copy()
    
    if len(validated) == 0:
        return pd.DataFrame()
    
    metrics_list = []
    
    for method in validated['method_pred'].dropna().unique():
        method_data = validated[validated['method_pred'] == method]
        
        # Conta TP, FP, FN, TN
        tp = len(method_data[method_data['comparison'] == 'TP'])
        fp = len(method_data[method_data['comparison'] == 'FP'])
        fn = len(method_data[method_data['comparison'] == 'FN'])
        tn = len(method_data[method_data['comparison'] == 'TN'])
        
        total = tp + fp + fn + tn
        
        if total > 0:
            # Calcola metriche
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / total if total > 0 else 0
            
            metrics_list.append({
                'method': method,
                'method_type': method_data['method_type'].iloc[0] if len(method_data) > 0 else 'Unknown',
                'total': total,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'avg_confidence': method_data['confidence'].mean() if 'confidence' in method_data.columns else None
            })
    
    return pd.DataFrame(metrics_list)


def get_field_names(df):
    """
    Estrae tutti i field_name unici dal dataset.
    
    Args:
        df: DataFrame con dati Lucy
        
    Returns:
        list: Lista di field_name ordinati
    """
    if 'field_name' not in df.columns:
        return []
    
    field_names = df['field_name'].dropna().unique().tolist()
    return sorted(field_names)


def filter_by_field_name(df, field_name):
    """
    Filtra il DataFrame per un specifico field_name.
    
    Args:
        df: DataFrame con dati Lucy
        field_name: Nome del campo da filtrare
        
    Returns:
        pd.DataFrame: DataFrame filtrato e preparato per quel campo
    """
    if 'field_name' not in df.columns:
        return df.copy()
    
    filtered_df = df[df['field_name'] == field_name].copy()
    
    # Prepara i dati se necessario (riapplica prepare_lucy_data se serve)
    if len(filtered_df) > 0:
        # Assicurati che le colonne necessarie siano presenti
        if 'datetime_sent' in filtered_df.columns:
            filtered_df['datetime_sent'] = pd.to_datetime(filtered_df['datetime_sent'])
            filtered_df['date'] = filtered_df['datetime_sent'].dt.date
            filtered_df['hour'] = filtered_df['datetime_sent'].dt.hour
            filtered_df['day_of_week'] = filtered_df['datetime_sent'].dt.day_name()
        
        # Crea flag per record validati
        filtered_df['is_validated'] = filtered_df['comparison'].notna()
        
        # Calcola accuratezza per record validati
        filtered_df['is_correct'] = filtered_df['comparison'].apply(lambda x: x == 'TP' if pd.notna(x) else None)
        
        # Categorizza metodi (ML vs Query)
        if 'method_pred' in filtered_df.columns:
            filtered_df['method_type'] = filtered_df['method_pred'].apply(categorize_method)
    
    return filtered_df


def calculate_metrics_by_field_and_method(df):
    """
    Calcola metriche aggregate per field_name e metodo.
    
    Args:
        df: DataFrame con dati validati
        
    Returns:
        pd.DataFrame: DataFrame con colonne: field_name, method, precision, recall, f1, accuracy, total
    """
    if 'field_name' not in df.columns:
        return pd.DataFrame()
    
    validated = df[df['is_validated']].copy()
    
    if len(validated) == 0:
        return pd.DataFrame()
    
    metrics_list = []
    
    for field_name in validated['field_name'].dropna().unique():
        field_data = validated[validated['field_name'] == field_name]
        
        for method in field_data['method_pred'].dropna().unique():
            method_data = field_data[field_data['method_pred'] == method]
            
            # Conta TP, FP, FN, TN
            tp = len(method_data[method_data['comparison'] == 'TP'])
            fp = len(method_data[method_data['comparison'] == 'FP'])
            fn = len(method_data[method_data['comparison'] == 'FN'])
            tn = len(method_data[method_data['comparison'] == 'TN'])
            
            total = tp + fp + fn + tn
            
            if total > 0:
                # Calcola metriche
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / total if total > 0 else 0
                
                metrics_list.append({
                    'field_name': field_name,
                    'method': method,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'total': total,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn
                })
    
    return pd.DataFrame(metrics_list)


def prepare_data(df):
    """
    Prepara i dati per l'analisi (pulizia, trasformazioni base).
    
    Args:
        df: DataFrame da preparare
        
    Returns:
        pd.DataFrame: DataFrame preparato
    """
    df = df.copy()
    # Aggiungi qui eventuali trasformazioni
    return df

