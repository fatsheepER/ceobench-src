import json
import os
from pathlib import Path
import subprocess
import sys


D0_STATE = '''
import json
from pathlib import Path
import sys
from numpy.random import default_rng
from saas_bench.config import BenchmarkConfig
from saas_bench.database import init_database
from saas_bench.simulation import Simulator
root = Path(sys.argv[1])
conn = init_database(root / 'world.db')
sim = Simulator(conn, BenchmarkConfig(seed=1045), default_rng(1045))
sim.initialize()
samples = {gid: sim._generate_customer_from_group(gid) for gid in sim.all_groups}
sim.save_rng_states()
state = json.loads(conn.execute("SELECT state_json FROM _rng_states WHERE name='all'").fetchone()[0])
rows = {table: [dict(row) for row in conn.execute('SELECT * FROM ' + table + ' ORDER BY 1')]
        for table in ('customers', 'group_parameters', 'group_info_levels', 'group_insight_snapshots')}
(root / 'state.json').write_text(json.dumps(dict(samples=samples, rng=state, rows=rows),
    sort_keys=True, default=lambda value: value.item()))
conn.close()
'''


def test_d0_group_state_is_identical_across_python_hash_seeds(tmp_path):
    states = []
    for hash_seed in ('0', '12345'):
        root = tmp_path / hash_seed
        root.mkdir()
        result = subprocess.run([sys.executable, '-c', D0_STATE, str(root)],
            env=dict(os.environ, PYTHONHASHSEED=hash_seed), capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        states.append((root / 'state.json').read_text())
    assert states[0] == states[1]
    state = json.loads(states[0])
    assert len(state['samples']) > 1 and len(state['rng']['_group_rngs']) > 1
    assert state['rng']['group_seed_hash'] == 'crc32'


def test_legacy_rng_hash_scheme_warns_and_restores_saved_states(make_initialized_sim, caplog):
    conn, sim, _ = make_initialized_sim()
    sim.save_rng_states()
    saved = json.loads(conn.execute("SELECT state_json FROM _rng_states WHERE name='all'").fetchone()[0])
    saved.pop('group_seed_hash')
    conn.execute("UPDATE _rng_states SET state_json=? WHERE name='all'", (json.dumps(saved),))
    sim.rng.random(20)
    assert sim.restore_rng_states()
    assert sim.rng.bit_generator.state == saved['rng']
    assert 'Checkpoint group RNG hash scheme is python-hash' in caplog.text
