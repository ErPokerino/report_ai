"""
Modulo per visualizzazioni specifiche dei dati Lucy.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configurazione stile
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_metrics_by_method(metrics_df, figsize=None):
    """
    Crea un grafico a barre delle metriche per metodo.
    
    Args:
        metrics_df: DataFrame con metriche per metodo
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        matplotlib.figure.Figure: Figura del grafico
    """
    if figsize is None:
        figsize = (12, 6)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepara dati
    methods = metrics_df['method'].values
    x = np.arange(len(methods))
    width = 0.2
    
    # Crea barre per ogni metrica
    ax.bar(x - 1.5*width, metrics_df['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x - 0.5*width, metrics_df['recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + 0.5*width, metrics_df['f1'], width, label='F1', alpha=0.8)
    ax.bar(x + 1.5*width, metrics_df['accuracy'], width, label='Accuracy', alpha=0.8)
    
    ax.set_xlabel('Metodo', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Metriche di Performance per Metodo', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix_by_method(df, method, figsize=None):
    """
    Crea una matrice di confusione per un metodo specifico.
    
    Args:
        df: DataFrame con dati validati
        method: Nome del metodo
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        matplotlib.figure.Figure: Figura del grafico
    """
    if figsize is None:
        figsize = (6, 5)
    
    method_data = df[(df['method_pred'] == method) & (df['is_validated'])]
    
    if len(method_data) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Nessun dato disponibile', ha='center', va='center')
        return fig
    
    # Conta TP, FP, FN, TN
    tp = len(method_data[method_data['comparison'] == 'TP'])
    fp = len(method_data[method_data['comparison'] == 'FP'])
    fn = len(method_data[method_data['comparison'] == 'FN'])
    tn = len(method_data[method_data['comparison'] == 'TN'])
    
    # Crea matrice di confusione
    cm = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo'],
                cbar_kws={"shrink": 0.8})
    ax.set_xlabel('Predetto', fontsize=11)
    ax.set_ylabel('Reale', fontsize=11)
    ax.set_title(f'Matrice di Confusione: {method}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_confidence_distribution(df, figsize=None):
    """
    Crea un grafico della distribuzione della confidence per metodo.
    
    Args:
        df: DataFrame con dati
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        matplotlib.figure.Figure: Figura del grafico
    """
    if figsize is None:
        figsize = (12, 6)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filtra solo record con confidence
    df_with_conf = df[df['confidence'].notna()].copy()
    
    if len(df_with_conf) > 0:
        # Box plot per metodo
        methods = df_with_conf['method_pred'].dropna().unique()
        data_to_plot = [df_with_conf[df_with_conf['method_pred'] == m]['confidence'].values 
                        for m in methods]
        
        bp = ax.boxplot(data_to_plot, labels=methods, patch_artist=True)
        
        # Colora i box
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel('Metodo', fontsize=11)
        ax.set_ylabel('Confidence', fontsize=11)
        ax.set_title('Distribuzione Confidence per Metodo', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_timeline_predictions(df, figsize=None):
    """
    Crea un grafico timeline delle predizioni nel tempo.
    
    Args:
        df: DataFrame con dati
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        matplotlib.figure.Figure: Figura del grafico
    """
    if figsize is None:
        figsize = (14, 6)
    
    # Raggruppa per data e metodo
    daily_counts = df.groupby([df['datetime_sent'].dt.date, 'method_pred']).size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['datetime_sent'])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot per ogni metodo
    methods = daily_counts['method_pred'].dropna().unique()
    for method in methods:
        method_data = daily_counts[daily_counts['method_pred'] == method]
        ax.plot(method_data['date'], method_data['count'], marker='o', label=method, linewidth=2)
    
    ax.set_xlabel('Data', fontsize=11)
    ax.set_ylabel('Numero di Predizioni', fontsize=11)
    ax.set_title('Timeline Predizioni per Metodo', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_accuracy_heatmap(df, figsize=None):
    """
    Crea una heatmap dell'accuratezza per country e metodo.
    
    Args:
        df: DataFrame con dati validati
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        matplotlib.figure.Figure: Figura del grafico
    """
    if figsize is None:
        figsize = (14, 10)
    
    validated = df[df['is_validated']].copy()
    
    if len(validated) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Nessun dato validato disponibile', ha='center', va='center', fontsize=14)
        return fig
    
    # Calcola accuracy per country e metodo
    accuracy_data = []
    for country in validated['country'].dropna().unique():
        for method in validated['method_pred'].dropna().unique():
            subset = validated[(validated['country'] == country) & 
                              (validated['method_pred'] == method)]
            if len(subset) > 0:
                correct = len(subset[subset['comparison'] == 'TP'])
                total = len(subset)
                accuracy = correct / total if total > 0 else 0
                accuracy_data.append({
                    'country': country,
                    'method': method,
                    'accuracy': accuracy,
                    'count': total
                })
    
    if len(accuracy_data) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Dati insufficienti per heatmap', ha='center', va='center', fontsize=14)
        return fig
    
    accuracy_df = pd.DataFrame(accuracy_data)
    pivot_table = accuracy_df.pivot(index='country', columns='method', values='accuracy')
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                vmin=0, vmax=1, ax=ax, cbar_kws={"shrink": 0.8},
                annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_xlabel('Metodo', fontsize=14, fontweight='bold')
    ax.set_ylabel('Country', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy per Country e Metodo', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    return fig


def plot_ml_vs_query_comparison(metrics_df, figsize=None):
    """
    Confronta performance ML vs Query-based.
    
    Args:
        metrics_df: DataFrame con metriche per metodo
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        matplotlib.figure.Figure: Figura del grafico
    """
    if figsize is None:
        figsize = (10, 6)
    
    # Raggruppa per tipo di metodo
    type_metrics = metrics_df.groupby('method_type').agg({
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean',
        'accuracy': 'mean',
        'total': 'sum'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(type_metrics))
    width = 0.2
    
    ax.bar(x - 1.5*width, type_metrics['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x - 0.5*width, type_metrics['recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + 0.5*width, type_metrics['f1'], width, label='F1', alpha=0.8)
    ax.bar(x + 1.5*width, type_metrics['accuracy'], width, label='Accuracy', alpha=0.8)
    
    ax.set_xlabel('Tipo di Metodo', fontsize=11)
    ax.set_ylabel('Score Medio', fontsize=11)
    ax.set_title('Confronto ML vs Query-based vs Other', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(type_metrics['method_type'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    return fig


def plot_method_usage(df, figsize=None):
    """
    Mostra la distribuzione dell'uso dei metodi.
    
    Args:
        df: DataFrame con dati
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        matplotlib.figure.Figure: Figura del grafico
    """
    if figsize is None:
        figsize = (10, 6)
    
    method_counts = df['method_pred'].value_counts()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(method_counts.index, method_counts.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Numero di Predizioni', fontsize=11)
    ax.set_ylabel('Metodo', fontsize=11)
    ax.set_title('Distribuzione Uso Metodi', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Aggiungi valori sulle barre
    for i, v in enumerate(method_counts.values):
        ax.text(v + 100, i, str(v), va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_metrics_by_field_name(metrics_df, figsize=None):
    """
    Crea un grafico a barre delle metriche aggregate per field_name.
    
    Args:
        metrics_df: DataFrame con metriche per field_name (da calculate_metrics_by_field_and_method)
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        matplotlib.figure.Figure: Figura del grafico
    """
    if figsize is None:
        figsize = (12, 6)
    
    if len(metrics_df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Nessun dato disponibile', ha='center', va='center')
        return fig
    
    # Aggrega per field_name
    field_metrics = metrics_df.groupby('field_name').agg({
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean',
        'accuracy': 'mean'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    field_names = field_metrics['field_name'].values
    x = np.arange(len(field_names))
    width = 0.2
    
    ax.bar(x - 1.5*width, field_metrics['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x - 0.5*width, field_metrics['recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + 0.5*width, field_metrics['f1'], width, label='F1', alpha=0.8)
    ax.bar(x + 1.5*width, field_metrics['accuracy'], width, label='Accuracy', alpha=0.8)
    
    ax.set_xlabel('Campo (field_name)', fontsize=11)
    ax.set_ylabel('Score Medio', fontsize=11)
    ax.set_title('Metriche di Performance per Campo', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(field_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    return fig


def plot_field_name_distribution(df, figsize=None):
    """
    Crea un grafico della distribuzione dei field_name.
    
    Args:
        df: DataFrame con dati
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        matplotlib.figure.Figure: Figura del grafico
    """
    if figsize is None:
        figsize = (10, 6)
    
    if 'field_name' not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Colonna field_name non presente', ha='center', va='center')
        return fig
    
    field_counts = df['field_name'].value_counts()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(field_counts.index, field_counts.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Numero di Record', fontsize=11)
    ax.set_ylabel('Campo (field_name)', fontsize=11)
    ax.set_title('Distribuzione Campi (field_name)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Aggiungi valori sulle barre
    for i, v in enumerate(field_counts.values):
        ax.text(v + max(field_counts.values) * 0.01, i, str(v), va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

