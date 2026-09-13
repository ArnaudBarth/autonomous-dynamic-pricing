"""Agents de tarification : baselines et agents d'apprentissage.

- ``agent_fixed``      — prix statique (baseline)
- ``agent_qtable``     — Q-Learning tabulaire (phase 1)
- ``agent_dqn`` / ``dqn`` — DQN "from scratch" TensorFlow (phase 2)

Tous exposent la même interface ``predict(obs, deterministic=True)`` (convention
Stable-Baselines3) -> comparables via ``evaluate.py``. ``agent_dqn`` n'est pas
importé ici pour éviter de charger TensorFlow sur un simple ``import pricing_agent``.

Les notebooks ajoutent ``../env`` et ``../pricing_agent`` au ``sys.path`` puis
importent les modules à plat (``from agent_fixed import FixedPriceAgent``).
"""
