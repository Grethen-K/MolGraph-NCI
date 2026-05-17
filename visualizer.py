# ============================================================
# 7. visualizer.py 3D-визуализация молекулярных структур с графом водородных связей.
#Выделение "подозрительных" контактов (раздел 6.3 ТЗ).
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx


def visualize_structure(G_full, title="Structure with H-bonds", save_path=None,
                         figsize=(12, 10), highlight_suspicious=True,
                         suspicious_threshold_dist=2.8, suspicious_threshold_angle=100):
    """
    Визуализирует структуру с ковалентными и ВС-рёбрами.

    Parameters
    ----------
    G_full : networkx.Graph
        Полный граф.
    title : str
        Заголовок графика.
    save_path : str or None
        Путь для сохранения PNG. Если None — показывает.
    figsize : tuple
        Размер фигуры.
    highlight_suspicious : bool
        Выделять ли подозрительные контакты (для варианта A).
    suspicious_threshold_dist : float
        Порог дистанции для подозрительных контактов (Å).
    suspicious_threshold_angle : float
        Порог угла для подозрительных контактов (°).
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Координаты вершин
    pos = {}
    elements = {}
    for node, data in G_full.nodes(data=True):
        pos[node] = data.get('coords', (0, 0, 0))
        elements[node] = data.get('element', 'X')

    # Цвета по элементам
    color_map = {
        'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
        'F': '#90E050', 'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F',
        'Br': '#A62929', 'I': '#940094', 'Fe': '#E06633', 'Pd': '#698598'
    }
    default_color = '#888888'

    # Рисуем ковалентные рёбра (серые)
    covalent_edges = [(u, v) for u, v, d in G_full.edges(data=True)
                      if d.get('type') == 'covalent']
    for u, v in covalent_edges:
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]
        ax.plot(x, y, z, color='gray', linewidth=1.0, alpha=0.6)

    # Рисуем HB-рёбра по типам
    hb_colors = {
        'HB_A': '#FF6B6B',   # Красный — слишком мягкий
        'HB_B': '#4ECDC4',   # Бирюзовый — химически осмысленный
        'HB_C': '#45B7D1'    # Синий — жёсткий
    }

    suspicious_edges = []
    for u, v, d in G_full.edges(data=True):
        etype = d.get('type', '')
        if etype.startswith('HB_'):
            color = hb_colors.get(etype, '#999999')
            linewidth = 2.5 if etype == 'HB_B' else 1.5
            alpha = 0.9 if etype == 'HB_B' else 0.6

            # Проверка на подозрительный контакт (только для A)
            is_suspicious = False
            if highlight_suspicious and etype == 'HB_A':
                dist = d.get('distance', 0.0)
                angle = d.get('angle', 0.0)
                if dist > suspicious_threshold_dist or angle < suspicious_threshold_angle:
                    is_suspicious = True
                    suspicious_edges.append((u, v, dist, angle))

            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            z = [pos[u][2], pos[v][2]]

            if is_suspicious:
                ax.plot(x, y, z, color='#FF00FF', linewidth=3.0, alpha=1.0,
                        linestyle='--', label='Suspicious HB_A')
            else:
                ax.plot(x, y, z, color=color, linewidth=linewidth, alpha=alpha)

    # Рисуем sigma-рёбра
    sigma_colors = {
        'sigma_A': '#FFA500',
        'sigma_B': '#32CD32',
        'sigma_C': '#8A2BE2'
    }
    for u, v, d in G_full.edges(data=True):
        etype = d.get('type', '')
        if etype.startswith('sigma_'):
            color = sigma_colors.get(etype, '#999999')
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            z = [pos[u][2], pos[v][2]]
            ax.plot(x, y, z, color=color, linewidth=2.0, linestyle=':', alpha=0.7)

    # Рисуем атомы
    for node, coords in pos.items():
        elem = elements[node]
        color = color_map.get(elem, default_color)
        size = 80 if elem == 'H' else 200
        ax.scatter(*coords, c=color, s=size, edgecolors='black', linewidths=0.5, alpha=0.9)

    # Подписи атомов
    for node, coords in pos.items():
        ax.text(coords[0], coords[1], coords[2], f'  {elements[node]}{node}',
                fontsize=7, alpha=0.7)

    # Легенда
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=1.0, label='Covalent'),
        Line2D([0], [0], color='#FF6B6B', lw=2.0, label='HB_A (soft)'),
        Line2D([0], [0], color='#4ECDC4', lw=2.5, label='HB_B (sensible)'),
        Line2D([0], [0], color='#45B7D1', lw=1.5, label='HB_C (strict)'),
        Line2D([0], [0], color='#FF00FF', lw=3.0, linestyle='--', label='Suspicious (A)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)

    # Добавляем информацию о подозрительных контактах
    if suspicious_edges:
        info_text = f"Suspicious contacts (HB_A): {len(suspicious_edges)}\\n"
        for u, v, dist, angle in suspicious_edges[:5]:
            info_text += f"  {u}-{v}: d={dist:.2f}Å, a={angle:.1f}°\\n"
        ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes,
                  fontsize=8, verticalalignment='bottom',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    return fig, ax


def visualize_compare_rules(G_A, G_B, G_C, title_prefix="",
                             save_path=None, figsize=(18, 5)):
    """
    Сравнительная визуализация трёх правил A/B/C на одной структуре.
    """
    fig = plt.figure(figsize=figsize)

    graphs = [('A (soft)', G_A, '#FF6B6B'),
              ('B (sensible)', G_B, '#4ECDC4'),
              ('C (strict)', G_C, '#45B7D1')]

    for idx, (label, G, color) in enumerate(graphs, 1):
        ax = fig.add_subplot(1, 3, idx, projection='3d')

        pos = {node: data.get('coords', (0, 0, 0))
               for node, data in G.nodes(data=True)}
        elements = {node: data.get('element', 'X')
                    for node, data in G.nodes(data=True)}

        # Ковалентные
        for u, v, d in G.edges(data=True):
            if d.get('type') == 'covalent':
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                z = [pos[u][2], pos[v][2]]
                ax.plot(x, y, z, color='gray', linewidth=0.8, alpha=0.5)

        # HB
        for u, v, d in G.edges(data=True):
            if d.get('type', '').startswith('HB_'):
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                z = [pos[u][2], pos[v][2]]
                ax.plot(x, y, z, color=color, linewidth=2.0, alpha=0.8)

        # Атомы
        for node, coords in pos.items():
            ax.scatter(*coords, c='black', s=50, alpha=0.8)

        # Статистика
        hb_edges = [e for e in G.edges(data=True)
                    if e[2].get('type', '').startswith('HB_')]
        ax.set_title(f"{title_prefix} {label}\\n{len(hb_edges)} H-bonds")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig


def visualize_hb_network(G_full, save_path=None, figsize=(10, 8)):
    """
    2D-визуализация только сети водородных связей (без ковалентного скелета).
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Извлекаем только HB-рёбра
    G_hb = nx.Graph()
    for u, v, d in G_full.edges(data=True):
        if d.get('type', '').startswith('HB_'):
            G_hb.add_edge(u, v, **d)

    if G_hb.number_of_edges() == 0:
        ax.text(0.5, 0.5, "No H-bonds found", ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=150)
        return fig, ax

    # Layout
    pos = nx.spring_layout(G_hb, k=2, iterations=50)

    # Цвета по типу HB
    edge_colors = []
    for u, v, d in G_hb.edges(data=True):
        etype = d.get('type', 'HB_B')
        if etype == 'HB_A':
            edge_colors.append('#FF6B6B')
        elif etype == 'HB_B':
            edge_colors.append('#4ECDC4')
        else:
            edge_colors.append('#45B7D1')

    nx.draw_networkx_nodes(G_hb, pos, node_color='lightblue',
                           node_size=300, ax=ax)
    nx.draw_networkx_labels(G_hb, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G_hb, pos, edge_color=edge_colors,
                           width=2.0, alpha=0.7, ax=ax)

    ax.set_title("H-bond Network")
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig, ax

