from mesa.discrete_space import CellAgent
from commons.commons import *

import random
from typing import Optional, List

PARTY = ["A", "B"]
POLARITYNEWS = [-1, 1]
VERACITYNEWS = [False, True]
PHI = 0.6
ALPHA = 0.12  # Aumentado para que las percepciones cambien más rápido

# Umbrales de conversión de agentes (más cercanos a 0 = conversiones más frecuentes)
THRESHOLD_TO_SKEPTIC = -0.10  # Si percepción hacia partido contrario es negativa
THRESHOLD_TO_SUSCEPTIBLE = 0.10  # Si percepción hacia partido contrario es positiva


class News:
    count = 0

    def __init__(self, id=None, party=None, polarity=None, veracity=None, credibility=None):  # f_k: credibilidad de la noticia
        super().__init__()  # Initialize News
        if id is None:
            id = News.count
        self.id = id
        News.count += 1
        self.party = party if party is not None else random.choice(PARTY)
        self.polarity = polarity if polarity is not None else random.choice(POLARITYNEWS)
        self.veracity = veracity if veracity is not None else random.choice(VERACITYNEWS)
        # Credibilidad de la noticia: alta si es verdadera, baja si es falsa
        if credibility is None:
            self.credibility = random.uniform(0.7, 0.9) if self.veracity else random.uniform(0.1, 0.3)
        else:
            self.credibility = credibility


class BOT(CellAgent):
    count = 0

    def __init__(self, model, id=None, initialnews=None, cell=None):
        super().__init__(model)  # Initialize News
        if id is None:
            id = BOT.count
        self.id = id
        BOT.count += 1
        self.initialnews = initialnews if initialnews is not None else []
        self.cell = cell

    def create_news(self):
        news = News(veracity=False)
        self.initialnews.append(news)

    def sendNews(self, news: News, radius: int = 1):
        # recorrer celdas vecinas
        for cell in self.cell.neighborhood:
            # recorrer agentes dentro de cada celda vecina
            for agent in cell.agents:
                # Solo enviar a Skeptic o Susceptible, NO a BOTs ni NewsReels
                agent_type = type(agent).__name__
                if agent_type not in ["Skeptic", "Susceptible"]:
                    continue
                if agent is not self:
                    agent.receiveNews(news)


class NewsReel(CellAgent):
    count = 0

    def __init__(self, model, id=None, initialnews=None, cell=None):
        super().__init__(model)  # Initialize News
        if id is None:
            id = NewsReel.count
        self.id = id
        NewsReel.count += 1
        self.initialnews = initialnews if initialnews is not None else []
        self.cell = cell

    def create_news(self):
        news = News(veracity=True)
        self.initialnews.append(news)

    def sendNews(self, news: News, radius: int = 1):
        # recorrer celdas vecinas
        for cell in self.cell.neighborhood:
            # recorrer agentes dentro de cada celda vecina
            for agent in cell.agents:
                # Solo enviar a Skeptic o Susceptible, NO a BOTs ni NewsReels
                agent_type = type(agent).__name__
                if agent_type not in ["Skeptic", "Susceptible"]:
                    continue
                if agent is not self:
                    agent.receiveNews(news)


class User(CellAgent):

    def __init__(self, model, id, partido: Optional[str] = None, credibility: Optional[float] = None, perception: Optional[dict[str, float]] = None, newsShared: Optional[List[News]] = None, newsReceived: Optional[List[News]] = None, cell=None):  # Partido político del agente (A o B)
        super().__init__(model)  # Initialize Agent

        self.id = id
        self.partido = partido if partido is not None else random.choice(PARTY)
        self.credibility = credibility
        self.perception = perception if perception is not None else {"A": 0.0, "B": 0.0}
        self.newsShared = newsShared if newsShared is not None else []
        self.newsReceived = newsReceived if newsReceived is not None else []

        self.newsReceivedIds = {n.id for n in self.newsReceived}
        self.newsSharedIds = {n.id for n in self.newsShared}
        self.newsExposureCount = {}

        self.cell = cell

    def computeShareProbability(self, news: News, w_m, w_f, w_c):
        """
        Calcula P_C según la propuesta:
        P_C = clamp(w_m·m_{i,j} + w_f·f_k + w_c·c_i, 0, 1)

        Donde m_{i,j} = (1 + polarity·p_j·p_i)/2
        p_i, p_j ∈ {+1, -1} representan partidos (A=+1, B=-1)
        """
        # Convertir partidos a valores numéricos: A=+1, B=-1
        p_i = 1 if self.partido == "A" else -1
        p_j = 1 if news.party == "A" else -1
        polarity = news.polarity  # +1 (pro) o -1 (contra)

        # Calcular indicador de alineamiento m_{i,j}
        m_ij = (1 + polarity * p_j * p_i) / 2.0

        # f_k: credibilidad de la noticia
        f_k = news.credibility

        # c_i: credibilidad del agente
        c_i = self.credibility

        # Calcular probabilidad de compartir
        P_C = clamp(w_m * m_ij + w_f * f_k + w_c * c_i, 0, 1)
        return P_C

    def receiveNews(self, news: News, sender: int = None):
        # Si ya vio la noticia: contabiliza exposición y no procesa de nuevo
        if news.id in self.newsReceivedIds:
            # opcional: contar exposiciones repetidas
            self.newsExposureCount[news.id] = self.newsExposureCount.get(news.id, 1) + 1
            return

        # primera vez que la ve
        self.newsReceived.append(news)
        self.newsReceivedIds.add(news.id)
        self.newsExposureCount[news.id] = 1

        # actualizar percepción la primera vez
        self.updatePerception(news)

        # verificar si debe convertirse a otro tipo
        new_type = self.checkConversion()
        if new_type is not None:
            self.convertTo(new_type)

        # decidir compartir: si decide, la envía (y registra que la compartió)
        try:
            share = self.shareDecision(news)
        except NotImplementedError:
            share = False

        if share:
            self.sendNews(news, sender=self.id)
            self.newsShared.append(news)
            self.newsSharedIds.add(news.id)

    def sendNews(self, news: News, sender: int = None, radius: int = 1):
        for cell in self.cell.neighborhood:
            for agent in cell.agents:
                # Solo enviar a Skeptic o Susceptible, NO a BOTs ni NewsReels
                agent_type = type(agent).__name__
                if agent_type not in ["Skeptic", "Susceptible"]:
                    continue

                if agent is not self:
                    # evita devolverla al emisor inmediato
                    if sender is not None and agent.id == sender:
                        continue
                    # evita enviar si el receptor ya vio la noticia
                    if hasattr(agent, "newsReceivedIds") and news.id in agent.newsReceivedIds:
                        continue

                    # enviar
                    agent.receiveNews(news, sender=self.id)

                    # Registrar la propagación SOLO después de envío exitoso
                    if hasattr(self.model, "news_propagation"):
                        self.model.news_propagation.append({"sender_id": self.id, "sender_type": self.__class__.__name__, "receiver_id": agent.id, "receiver_type": agent.__class__.__name__, "news_id": news.id, "news_party": news.party, "news_veracity": news.veracity})

    def convertTo(self, new_type):
        """
        Convierte el agente al nuevo tipo especificado.
        Mantiene todos los atributos (percepción, credibilidad, noticias) pero cambia el comportamiento.
        """
        old_type_name = self.__class__.__name__
        new_type_name = new_type.__name__

        # Ajustar credibilidad según el nuevo tipo
        if new_type == Skeptic:
            # Al volverse escéptico, reduce su credibilidad
            self.credibility = clamp(self.credibility * 0.5, 0.1, 0.3)
            # Incrementar contador en el modelo
            if hasattr(self.model, "conversions_to_skeptic"):
                self.model.conversions_to_skeptic += 1
        elif new_type == Susceptible:
            # Al volverse susceptible, aumenta su credibilidad
            self.credibility = clamp(self.credibility * 2.0, 0.6, 0.9)
            # Incrementar contador en el modelo
            if hasattr(self.model, "conversions_to_susceptible"):
                self.model.conversions_to_susceptible += 1

        # Obtener posición en el grid
        cell_pos = self.cell.coordinate if hasattr(self.cell, "coordinate") else "desconocida"

        # Obtener percepción actual
        other_party = "B" if self.partido == "A" else "A"
        perception = self.perception[other_party]

        # Cambiar la clase del objeto
        self.__class__ = new_type

        # Registrar la conversión en el modelo para mostrar después
        if hasattr(self.model, "converted_agents"):
            self.model.converted_agents.append({"id": self.id, "old_type": old_type_name, "new_type": new_type_name, "partido": self.partido, "position": cell_pos, "perception": perception, "new_credibility": self.credibility})

        # PAUSAR el modelo cuando hay conversión
        if hasattr(self.model, "running"):
            self.model.running = False

        print(f"\n{'='*60}")
        print(f"🔄 CONVERSIÓN DE AGENTE DETECTADA")
        print(f"{'='*60}")
        print(f"  >> Agente ID: {self.id}")
        print(f"  >> Cambio: {old_type_name} → {new_type_name}")
        print(f"  >> Partido: {self.partido}")
        print(f"  >> Posición: {cell_pos}")
        print(f"  >> Percepción hacia partido contrario: {perception:.3f}")
        print(f"  >> Nueva credibilidad: {self.credibility:.3f}")
        print(f"  >> Noticias recibidas: {len(self.newsReceived)}")
        print(f"  >> Noticias compartidas: {len(self.newsShared)}")
        print(f"{'='*60}")
        print(f"⏸️  SIMULACIÓN PAUSADA - Presiona STEP o PLAY para continuar")
        print(f"{'='*60}\n")

    def shareDecision(self, news: News) -> bool:
        """Abstracto: devolver True si decide compartir (según la noticia)."""
        raise NotImplementedError

    def updatePerception(self, news: News):
        """
        Actualiza la percepción según la propuesta:
        P_{i,t} = clamp(P_{i,t-1} + x_j·α·polarity·c_i, -1, 1)

        Donde x_j = 1 si la noticia es del partido contrario, 0 si es del mismo
        """
        # Determinar si la noticia es del partido contrario
        x_j = 1 if news.party != self.partido else 0

        # Calcular incremento de percepción
        delta = x_j * ALPHA * news.polarity * self.credibility

        # Actualizar percepción del partido de la noticia
        old_perception = self.perception[news.party]
        new_perception = clamp(old_perception + delta, -1.0, 1.0)
        self.perception[news.party] = roundto(new_perception)

    def checkConversion(self):
        """
        Verifica si el agente debe convertirse a otro tipo basándose en su percepción.
        Retorna el tipo al que debe convertirse o None si no debe cambiar.
        """
        # Obtener percepción hacia el partido contrario
        other_party = "B" if self.partido == "A" else "A"
        perception_to_other = self.perception[other_party]

        # Determinar si debe convertirse
        if isinstance(self, Susceptible):
            # Susceptible se vuelve Skeptic si tiene percepción muy negativa hacia el partido contrario
            if perception_to_other <= THRESHOLD_TO_SKEPTIC:
                return Skeptic
        elif isinstance(self, Skeptic):
            # Skeptic se vuelve Susceptible si tiene percepción muy positiva hacia el partido contrario
            if perception_to_other >= THRESHOLD_TO_SUSCEPTIBLE:
                return Susceptible

        return None


class Susceptible(User):
    def __init__(self, model, id=None, credibility=None, **kwargs):
        if id is None:
            id = len(model.agents_by_type[Susceptible]) if Susceptible in model.agents_by_type else 0
        super().__init__(model, id, credibility=credibility, **kwargs)
        if self.credibility is None:
            self.credibility = random.uniform(0.6, 0.9)

    def shareDecision(self, news: News) -> bool:
        w1 = 0.1
        w2 = 0.3
        w3 = 0.6
        pc = self.computeShareProbability(news, w1, w2, w3)
        return random.random() < pc

    def updatePerception(self, news: News):
        """
        Susceptible actualiza percepción con TODAS las noticias.
        Usa la fórmula: P_{i,t} = clamp(P_{i,t-1} + x_j·α·polarity·c_i, -1, 1)
        """
        # Determinar si la noticia es del partido contrario
        x_j = 1 if news.party != self.partido else 0

        # Calcular incremento de percepción
        delta = x_j * ALPHA * news.polarity * self.credibility

        # Actualizar percepción del partido de la noticia
        old_perception = self.perception[news.party]
        new_perception = clamp(old_perception + delta, -1.0, 1.0)
        self.perception[news.party] = roundto(new_perception)


class Skeptic(User):
    def __init__(self, model, id=None, credibility=None, **kwargs):
        if id is None:
            id = len(model.agents_by_type[Skeptic]) if Skeptic in model.agents_by_type else 0
        super().__init__(model, id, credibility=credibility, **kwargs)
        if self.credibility is None:
            self.credibility = random.uniform(0.1, 0.3)

    def shareDecision(self, news: News) -> bool:
        w1 = 0.3
        w2 = 0.6
        w3 = 0.1
        if news.veracity == False:
            return False
        else:
            pc = self.computeShareProbability(news, w1, w2, w3)
            return random.random() < pc

    def updatePerception(self, news: News):
        """
        Skeptic SOLO actualiza percepción con noticias VERDADERAS.
        Usa la fórmula: P_{i,t} = clamp(P_{i,t-1} + x_j·α·polarity·c_i, -1, 1)
        pero con factor=0 si la noticia es falsa.
        """
        # Solo actualiza si la noticia es verdadera
        if not news.veracity:
            return

        # Determinar si la noticia es del partido contrario
        x_j = 1 if news.party != self.partido else 0

        # Calcular incremento de percepción
        delta = x_j * ALPHA * news.polarity * self.credibility

        # Actualizar percepción del partido de la noticia
        old_perception = self.perception[news.party]
        new_perception = clamp(old_perception + delta, -1.0, 1.0)
        self.perception[news.party] = roundto(new_perception)
