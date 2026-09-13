"""Test de fumée : vérifie que le simulateur et l'interface Gym tournent."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "env"))

import numpy as np

from gym_wrapper import TicketEnv, discretize_observation
from market import TicketMarketModel
from refdata import ArtistProfile, VenueProfile, load_artists


def test_market_direct():
    m = TicketMarketModel(artist="Powerwolf", venue="Hallenstadion", horizon=100, seed=0)
    m.current_price = 60.0
    while not m.done:
        m.step()
    assert m.tickets_sold_total > 0
    assert m.tickets_sold_total <= m.initial_tickets
    assert 1 <= len(m.sales_history) <= m.initial_time      # peut se vendre avant la date
    print(
        f"  market direct: {m.artist.name} pop={m.popularity_index:.1f} "
        f"vendus={m.tickets_sold_total}/{m.initial_tickets} fill={m.fill_rate:.1%}"
    )


def test_domain_randomization():
    seen = set()
    pops = []
    for s in range(30):
        m = TicketMarketModel(seed=s)
        seen.add(m.artist.name)
        pops.append(m.popularity_index)
    assert len(seen) >= 4, f"DR faible, artistes vus: {seen}"
    print(f"  DR: {len(seen)} artistes distincts, popularité {min(pops):.1f}..{max(pops):.1f}")


def test_gym_api():
    env = TicketEnv(horizon=80)
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs), obs
    total_r = 0.0
    steps = 0
    while True:
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        assert env.observation_space.contains(obs), obs
        total_r += r
        steps += 1
        if term or trunc:
            break
    assert steps <= 80
    print(
        f"  gym: {steps} pas, retour={total_r:.2f}, CA={info['revenue_total']:.0f} CHF, "
        f"fill={info['fill_rate']:.1%} ({info['artist']} @ {info['venue']})"
    )


def test_reproducibility():
    env = TicketEnv()
    o1, _ = env.reset(seed=7)
    traj1 = [env.step(3)[1] for _ in range(20)]
    o2, _ = env.reset(seed=7)
    traj2 = [env.step(3)[1] for _ in range(20)]
    assert np.allclose(o1, o2)
    assert np.allclose(traj1, traj2), "reset(seed) non reproductible"
    print("  reproductibilité: OK")


def test_price_monotonicity():
    """Prix plus élevé -> moins de ventes (toutes choses égales par ailleurs)."""
    sold = {}
    for action in (0, 4, 9):
        env = TicketEnv(domain_randomize=False)
        env.reset(seed=1, options={"artist": "Sabaton", "venue": "The Hall"})
        while not env.step(action)[2]:
            pass
        sold[action] = env.market.tickets_sold_total
    print(f"  monotonie prix: ventes par action {sold}")
    assert sold[0] >= sold[4] >= sold[9], sold


def test_discretize():
    from gym_wrapper import STATE_BINS

    for _ in range(1000):
        state = discretize_observation(np.random.rand(7).astype(np.float32))
        assert len(state) == len(STATE_BINS)
        assert all(0 <= s < b for s, b in zip(state, STATE_BINS))
    assert discretize_observation(np.ones(7)) == tuple(b - 1 for b in STATE_BINS)
    assert discretize_observation(np.zeros(7)) == tuple(0 for _ in STATE_BINS)
    print("  discretize_observation: bornes OK")


if __name__ == "__main__":
    print("artists.csv:", len(load_artists()), "artistes")
    for fn in [
        test_market_direct,
        test_domain_randomization,
        test_gym_api,
        test_reproducibility,
        test_price_monotonicity,
        test_discretize,
    ]:
        print(f"- {fn.__name__}")
        fn()
    print("\nOK - tout passe.")
