"""
Modulo per visualizzazioni specifiche dei dati Lucy.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Configurazione stile
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Costanti dimensioni figure
DEFAULT_FIGSIZE = (10, 6)
METRICS_FIGSIZE = (12, 6)
HEATMAP_FIGSIZE = (14, 10)
CONFUSION_FIGSIZE = (6, 5)
TIMELINE_FIGSIZE = (14, 6)

# Costanti stile
BAR_ALPHA = 0.8
GRID_ALPHA = 0.3
BAR_WIDTH = 0.2
Y_LIM_MAX = 1.1


def _setup_figure(
    figsize: Tuple[float, float],
    title: str,
    xlabel: str,
    ylabel: str,
    grid: bool = True,
    grid_axis: str = 'y'
) -> Tuple[Figure, Axes]:
    """
    Setup comune per figure matplotlib.
    
    Args:
        figsize: Dimensioni figura (width, height)
        title: Titolo del grafico
        xlabel: Etichetta asse X
        ylabel: Etichetta asse Y
        grid: Se True, mostra griglia
        grid_axis: Asse per griglia ('y', 'x', o 'both')
        
    Returns:
        Tupla (figura, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    if grid:
        ax.grid(True, alpha=GRID_ALPHA, axis=grid_axis)
    return fig, ax


def _plot_metrics_bars(
    ax: Axes,
    x: np.ndarray,
    metrics_data: pd.DataFrame,
    labels: List[str],
    width: float = BAR_WIDTH
) -> None:
    """
    Helper per creare barre metriche multiple.
    
    Args:
        ax: Axes su cui disegnare
        x: Posizioni X per le barre
        metrics_data: DataFrame con colonne per ogni metrica
        labels: Lista di etichette per le metriche
        width: Larghezza delle barre
    """
    metric_names = ['precision', 'recall', 'f1', 'accuracy']
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    
    for metric, offset, label in zip(metric_names, offsets, labels):
        if metric in metrics_data.columns:
            ax.bar(x + offset, metrics_data[metric], width, label=label, alpha=BAR_ALPHA)


def plot_metrics_by_method(metrics_df: pd.DataFrame, figsize: Optional[Tuple[float, float]] = None) -> Figure:
    """
    Crea un grafico a barre delle metriche per metodo.
    
    Args:
        metrics_df: DataFrame con metriche per metodo
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        Figura del grafico
    """
    if figsize is None:
        figsize = METRICS_FIGSIZE
    
    fig, ax = _setup_figure(
        figsize,
        'Metriche di Performance per Metodo',
        'Metodo',
        'Score'
    )
    
    # Prepara dati
    methods = metrics_df['method'].values
    x = np.arange(len(methods))
    
    # Crea barre per ogni metrica usando helper
    _plot_metrics_bars(ax, x, metrics_df, ['Precision', 'Recall', 'F1', 'Accuracy'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, Y_LIM_MAX])
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix_by_method(
    df: pd.DataFrame,
    method: str,
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    """
    Crea una matrice di confusione per un metodo specifico.
    
    Args:
        df: DataFrame con dati validati
        method: Nome del metodo
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        Figura del grafico
    """
    if figsize is None:
        figsize = CONFUSION_FIGSIZE
    
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
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Negativo', 'Positivo'],
        yticklabels=['Negativo', 'Positivo'],
        cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel('Predetto', fontsize=11)
    ax.set_ylabel('Reale', fontsize=11)
    ax.set_title(f'Matrice di Confusione: {method}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_confidence_distribution(
    df: pd.DataFrame,
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    """
    Crea un grafico della distribuzione della confidence per metodo.
    
    Args:
        df: DataFrame con dati
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        Figura del grafico
    """
    if figsize is None:
        figsize = METRICS_FIGSIZE
    
    fig, ax = _setup_figure(
        figsize,
        'Distribuzione Confidence per Metodo',
        'Metodo',
        'Confidence'
    )
    
    # Filtra solo record con confidence
    df_with_conf = df[df['confidence'].notna()].copy()
    
    if len(df_with_conf) > 0:
        # Box plot per metodo
        methods = df_with_conf['method_pred'].dropna().unique()
        data_to_plot = [
            df_with_conf[df_with_conf['method_pred'] == m]['confidence'].values 
            for m in methods
        ]
        
        bp = ax.boxplot(data_to_plot, labels=methods, patch_artist=True)
        
        # Colora i box
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_timeline_predictions(
    df: pd.DataFrame,
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    """
    Crea un grafico timeline delle predizioni nel tempo.
    
    Args:
        df: DataFrame con dati
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        Figura del grafico
    """
    if figsize is None:
        figsize = TIMELINE_FIGSIZE
    
    # Raggruppa per data e metodo
    daily_counts = df.groupby([df['datetime_sent'].dt.date, 'method_pred']).size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['datetime_sent'])
    
    fig, ax = _setup_figure(
        figsize,
        'Timeline Predizioni per Metodo',
        'Data',
        'Numero di Predizioni',
        grid_axis='both'
    )
    
    # Plot per ogni metodo
    methods = daily_counts['method_pred'].dropna().unique()
    for method in methods:
        method_data = daily_counts[daily_counts['method_pred'] == method]
        ax.plot(method_data['date'], method_data['count'], marker='o', label=method, linewidth=2)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_accuracy_heatmap(
    df: pd.DataFrame,
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    """
    Crea una heatmap dell'accuratezza per country e metodo.
    
    Args:
        df: DataFrame con dati validati
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        Figura del grafico
    """
    if figsize is None:
        figsize = HEATMAP_FIGSIZE
    
    validated = df[df['is_validated']].copy() if 'is_validated' in df.columns else pd.DataFrame()
    
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
    sns.heatmap(
        pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
        vmin=0, vmax=1, ax=ax, cbar_kws={"shrink": 0.8},
        annot_kws={'size': 12, 'weight': 'bold'}
    )
    ax.set_xlabel('Metodo', fontsize=14, fontweight='bold')
    ax.set_ylabel('Country', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy per Country e Metodo', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    return fig


def plot_ml_vs_query_comparison(
    metrics_df: pd.DataFrame,
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    """
    Confronta performance ML vs Query-based.
    
    Args:
        metrics_df: DataFrame con metriche per metodo
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        Figura del grafico
    """
    if figsize is None:
        figsize = DEFAULT_FIGSIZE
    
    # Raggruppa per tipo di metodo
    type_metrics = metrics_df.groupby('method_type').agg({
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean',
        'accuracy': 'mean',
        'total': 'sum'
    }).reset_index()
    
    fig, ax = _setup_figure(
        figsize,
        'Confronto ML vs Query-based vs Other',
        'Tipo di Metodo',
        'Score Medio'
    )
    
    x = np.arange(len(type_metrics))
    
    # Usa helper per barre metriche
    _plot_metrics_bars(ax, x, type_metrics, ['Precision', 'Recall', 'F1', 'Accuracy'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(type_metrics['method_type'])
    ax.legend()
    ax.set_ylim([0, Y_LIM_MAX])
    
    plt.tight_layout()
    return fig


def plot_method_usage(
    df: pd.DataFrame,
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    """
    Mostra la distribuzione dell'uso dei metodi.
    
    Args:
        df: DataFrame con dati
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        Figura del grafico
    """
    if figsize is None:
        figsize = DEFAULT_FIGSIZE
    
    method_counts = df['method_pred'].value_counts()
    
    fig, ax = _setup_figure(
        figsize,
        'Distribuzione Uso Metodi',
        'Numero di Predizioni',
        'Metodo',
        grid_axis='x'
    )
    
    ax.barh(method_counts.index, method_counts.values, color='steelblue', alpha=BAR_ALPHA)
    
    # Aggiungi valori sulle barre
    for i, v in enumerate(method_counts.values):
        ax.text(v + 100, i, str(v), va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_metrics_by_field_name(
    metrics_df: pd.DataFrame,
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    """
    Crea un grafico a barre delle metriche aggregate per field_name.
    
    Args:
        metrics_df: DataFrame con metriche per field_name (da calculate_metrics_by_field_and_method)
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        Figura del grafico
    """
    if figsize is None:
        figsize = METRICS_FIGSIZE
    
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
    
    fig, ax = _setup_figure(
        figsize,
        'Metriche di Performance per Campo',
        'Campo (field_name)',
        'Score Medio'
    )
    
    field_names = field_metrics['field_name'].values
    x = np.arange(len(field_names))
    
    # Usa helper per barre metriche
    _plot_metrics_bars(ax, x, field_metrics, ['Precision', 'Recall', 'F1', 'Accuracy'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(field_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, Y_LIM_MAX])
    
    plt.tight_layout()
    return fig


def plot_field_name_distribution(
    df: pd.DataFrame,
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    """
    Crea un grafico della distribuzione dei field_name.
    
    Args:
        df: DataFrame con dati
        figsize: Tuple (width, height) per le dimensioni
        
    Returns:
        Figura del grafico
    """
    if figsize is None:
        figsize = DEFAULT_FIGSIZE
    
    if 'field_name' not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Colonna field_name non presente', ha='center', va='center')
        return fig
    
    field_counts = df['field_name'].value_counts()
    
    fig, ax = _setup_figure(
        figsize,
        'Distribuzione Campi (field_name)',
        'Numero di Record',
        'Campo (field_name)',
        grid_axis='x'
    )
    
    ax.barh(field_counts.index, field_counts.values, color='steelblue', alpha=BAR_ALPHA)
    
    # Aggiungi valori sulle barre
    for i, v in enumerate(field_counts.values):
        ax.text(v + max(field_counts.values) * 0.01, i, str(v), va='center', fontsize=9)
    
    plt.tight_layout()
    return fig
