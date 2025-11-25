from model import SocialNetworkModel
from agents import Susceptible, Skeptic, BOT, NewsReel
from mesa.experimental.devs import ABMSimulator
from mesa.visualization import (
    CommandConsole,
    Slider,
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle
import solara
from matplotlib.figure import Figure


def social_network_portrayal(agent):
    if agent is None:
        return

    portrayal = AgentPortrayalStyle(size=50, marker="o", zorder=2, edgecolors="black")

    if isinstance(agent, Susceptible):
        portrayal.update(("color", "red"))
    elif isinstance(agent, Skeptic):
        portrayal.update(("color", "blue"))
    elif isinstance(agent, BOT):
        portrayal.update(("color", "black"))
    elif isinstance(agent, NewsReel):
        portrayal.update(("color", "white"))

    return portrayal


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "width": Slider("Grid Width", 20, 5, 50),
    "height": Slider("Grid Height", 20, 5, 50),
    "n_susceptible": Slider("Initial Susceptible Users", 70, 1, 200),
    "n_skeptic": Slider("Initial Skeptic Users", 70, 1, 200),
}


def post_process_lines(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))


# Crear los componentes de gráficos una sola vez
perception_plot = make_plot_component({"AvgPerception_Skeptic": "tab:blue", "AvgPerception_Susceptible": "tab:red"})
shared_news_plot = make_plot_component({"TrueNewsShared": "tab:blue", "FalseNewsShared": "tab:red"})


# Variable global para rastrear el modelo actual
_current_model_ref = {"model": None, "last_steps": 0}


@solara.component
def SpaceWithArrows(model):
    """Componente personalizado para el espacio con flechas dinámicas"""
    # Forzar re-render monitoreando steps del modelo global
    current_model = model.value if hasattr(model, "value") else model

    # Guardar referencia global
    _current_model_ref["model"] = current_model

    # Verificar que model es una instancia
    if not hasattr(current_model, "width"):
        return solara.Text("Waiting for model initialization...")

    # CLAVE: Usar un state local que se actualiza externamente
    render_trigger, set_render_trigger = solara.use_state(0)

    # Effect que monitorea cambios
    def monitor_changes():
        import time
        import threading

        def check_loop():
            last_steps = _current_model_ref["last_steps"]
            while True:
                time.sleep(0.3)
                m = _current_model_ref["model"]
                if m and hasattr(m, "steps"):
                    current_steps = m.steps
                    if current_steps != last_steps:
                        _current_model_ref["last_steps"] = current_steps
                        set_render_trigger(current_steps)
                        last_steps = current_steps

        thread = threading.Thread(target=check_loop, daemon=True)
        thread.start()

    solara.use_effect(monitor_changes, [])

    # Leer propiedades del modelo
    steps = current_model.steps
    width = current_model.width
    height = current_model.height
    news_propagation = list(current_model.news_propagation) if hasattr(current_model, "news_propagation") else []

    # DEBUG
    print(f"\n=== RENDER Steps: {steps}, Trigger: {render_trigger}, Propagations: {len(news_propagation)} ===")

    # Crear figura
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Dibujar grid
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_xticks([x + 0.5 for x in range(width)])
    ax.set_yticks([y + 0.5 for y in range(height)])
    ax.grid(True, which="both", color="lightgray", linewidth=0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Capturar posiciones y colores de agentes
    agents_data = []
    agent_positions = {}
    for agent in current_model.agents:
        if hasattr(agent, "cell") and hasattr(agent.cell, "coordinate"):
            portrayal = social_network_portrayal(agent)
            if portrayal:
                x, y = agent.cell.coordinate
                agents_data.append(
                    {
                        "x": x,
                        "y": y,
                        "size": portrayal.size if hasattr(portrayal, "size") else 50,
                        "color": portrayal.color if hasattr(portrayal, "color") else "gray",
                        "marker": portrayal.marker if hasattr(portrayal, "marker") else "o",
                        "edgecolors": portrayal.edgecolors if hasattr(portrayal, "edgecolors") else "black",
                        "zorder": portrayal.zorder if hasattr(portrayal, "zorder") else 2,
                    }
                )
                agent_positions[agent.id] = (x, y)

    # Dibujar agentes
    for agent_data in agents_data:
        ax.scatter(
            agent_data["x"],
            agent_data["y"],
            s=agent_data["size"],
            c=agent_data["color"],
            marker=agent_data["marker"],
            edgecolors=agent_data["edgecolors"],
            zorder=agent_data["zorder"],
        )

    # Dibujar flechas de propagación
    for prop in news_propagation:
        sender_id = prop["sender_id"]
        receiver_id = prop["receiver_id"]

        if sender_id in agent_positions and receiver_id in agent_positions:
            sender_pos = agent_positions[sender_id]
            receiver_pos = agent_positions[receiver_id]

            arrow_color = "green" if prop["news_veracity"] else "red"

            ax.annotate(
                "",
                xy=receiver_pos,
                xytext=sender_pos,
                arrowprops=dict(
                    arrowstyle="->",
                    color=arrow_color,
                    alpha=0.7,
                    lw=2.5,
                    shrinkA=8,
                    shrinkB=8,
                ),
            )

    solara.FigureMatplotlib(fig, dependencies=[steps, len(agents_data), len(news_propagation), width, height, id(current_model)])


@solara.component
def Page():
    # Crear modelo inicial
    simulator = solara.use_memo(lambda: ABMSimulator(), dependencies=[])
    model_state, set_model_state = solara.use_state(None)

    # Inicializar modelo una vez
    def init_model():
        if model_state is None:
            m = SocialNetworkModel(simulator=simulator)
            set_model_state(m)
            return m
        return model_state

    initial_model = solara.use_memo(init_model, dependencies=[model_state])

    # Forzar actualización cuando el modelo cambia
    # Esto es un workaround - incrementar counter para forzar re-render de componentes hijos
    update_counter = solara.use_reactive(0)

    def increment_counter():
        if initial_model and hasattr(initial_model, "steps"):
            update_counter.value = initial_model.steps

    # Polling: revisar el modelo periódicamente
    def poll_model():
        import threading
        import time

        def poll():
            while True:
                time.sleep(0.5)
                if initial_model:
                    increment_counter()

        thread = threading.Thread(target=poll, daemon=True)
        thread.start()

    solara.use_effect(poll_model, [])

    if initial_model is None:
        return solara.Text("Initializing...")

    # Pasar el counter como parte de un wrapper para forzar re-render
    with solara.Column():
        solara.Text(f"Update counter: {update_counter.value}")
        SolaraViz(
            initial_model,
            components=[SpaceWithArrows, perception_plot, shared_news_plot, CommandConsole],
            model_params=model_params,
            name="Social Network Simulation",
        )


page = Page
