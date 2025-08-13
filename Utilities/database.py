import sqlite3
import os
import secrets
import string
import shutil
from datetime import datetime

DB_NAME = "job_database.db"
TABLE_NAME = "jobs"
STEPS = ["S0", "S1", "S2", "S3", "S4"]

def secure_random_string(length=12):
    """Generate a secure random alphanumeric string (default length = 12)."""
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))

def init_db():
    """Initialize the SQLite database and create the jobs table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            job_id TEXT PRIMARY KEY,
            MEname TEXT,
            created_at TEXT,
            S0 TEXT,
            S1 TEXT,
            S2 TEXT,
            S3 TEXT,
            S4 TEXT
        )
    """)
    conn.commit()
    conn.close()

def create_new_job_folder(MEname, base_path="outputs"):
    """
    Create a new job entry with a secure random job_id, MEname, and created_at timestamp.
    Also creates a corresponding output folder.
    """
    job_id = secure_random_string()
    job_path = os.path.join(base_path, MEname+"-"+job_id) + os.sep
    os.makedirs(job_path, exist_ok=True)

    created_at = datetime.utcnow().isoformat()

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(f"""
        INSERT INTO {TABLE_NAME} (MEname, job_id, created_at)
        VALUES (?, ?, ?)
    """, (MEname, job_id, created_at))
    conn.commit()
    conn.close()

    return job_id, job_path

def update_step(job_id, step_name, output_path):
    """
    Update the output path of a specific step (S0–S4) for the given job_id.
    If the step is already filled, a new job is cloned with previous steps,
    and the update is applied to the new job instead.
    Returns the job_id and job_path where the update was applied.
    """
    assert step_name in STEPS, f"{step_name} is not a valid step."

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Check if the step is already filled
    c.execute(f"SELECT {step_name} FROM {TABLE_NAME} WHERE job_id = ?", (job_id,))
    row = c.fetchone()
    if row is None:
        raise ValueError(f"No job found with ID {job_id}")

    if row[0] is not None:
        # Step already filled: clone and update the new job
        new_job_id, new_job_path = clone_job_with_previous_steps(step_name, job_id)
        return new_job_id, new_job_path

    # Step not filled: update current job
    c.execute(f"""
        UPDATE {TABLE_NAME}
        SET {step_name} = ?
        WHERE job_id = ?
    """, (output_path, job_id))
    conn.commit()

    # Get job_path
    c.execute(f"SELECT MEname FROM {TABLE_NAME} WHERE job_id = ?", (job_id,))
    MEname = c.fetchone()[0]
    job_path = os.path.join("outputs", job_id, MEname)

    conn.close()
    return job_id, job_path
    
def clone_job_with_steps(from_job_id, steps_to_copy):
    """
    Clone selected steps from an existing job and create a new job entry and folder.
    Only the specified steps are copied; others remain NULL.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Fetch MEname from original job
    c.execute(f"SELECT MEname FROM {TABLE_NAME} WHERE job_id = ?", (from_job_id,))
    row = c.fetchone()
    if row is None:
        raise ValueError(f"No job found with ID {from_job_id}")
    MEname = row[0]
    conn.close()

    # Create new job and folder
    new_job_id, job_path = create_new_job_folder(MEname)

    # Reopen connection
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Fetch values for requested steps
    cols = ", ".join(steps_to_copy)
    c.execute(f"SELECT {cols} FROM {TABLE_NAME} WHERE job_id = ?", (from_job_id,))
    values = c.fetchone()

    # NON fare INSERT, ma UPDATE per il job appena creato:
    set_clause = ", ".join([f"{col} = ?" for col in steps_to_copy])
    c.execute(f"""
        UPDATE {TABLE_NAME}
        SET {set_clause}
        WHERE job_id = ?
    """, (*values, new_job_id))

    conn.commit()
    conn.close()

    return new_job_id, job_path
    
def clone_job_with_previous_steps(current_step, from_job_id):
    """
    Clone all steps prior to the current step from an existing job.
    For example, if current_step = 'S3', then only S1 and S2 are cloned.
    """
    if current_step not in STEPS:
        raise ValueError(f"{current_step} is not a valid step. Allowed steps: {STEPS}")
    step_index = STEPS.index(current_step)
    steps_to_copy = STEPS[:step_index]
    return clone_job_with_steps(from_job_id, steps_to_copy)

# === Query functions ===

def get_job_info(job_id):
    """
    Return a dictionary with all columns for the given job_id.
    If job_id does not exist, return None.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(f"SELECT * FROM {TABLE_NAME} WHERE job_id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return None
    return dict(zip(["job_id", "MEname", "created_at"] + STEPS, row))

def get_step_output(job_id, step_name):
    """
    Return the output path of a specific step for the given job_id.
    Returns None if not set or if job doesn't exist.
    """
    assert step_name in STEPS, f"{step_name} is not a valid step."
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(f"SELECT {step_name} FROM {TABLE_NAME} WHERE job_id = ?", (job_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def get_job_status(job_id):
    """
    Return a dictionary showing whether each step has been completed (True/False).
    """
    job_info = get_job_info(job_id)
    if job_info is None:
        return None
    return {step: (job_info[step] is not None) for step in STEPS}

def list_recent_jobs(limit=5):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(f"""
        SELECT job_id, MEname, created_at, {", ".join(STEPS)}
        FROM {TABLE_NAME}
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()

    jobs = []
    print(f"\nUltimi {limit} job nel database:")
    print("=" * 80)
    for row in rows:
        job_id, MEname, created_at, *steps = row
        status = {step: bool(s) for step, s in zip(STEPS, steps)}
        jobs.append({
            "job_id": job_id,
            "MEname": MEname,
            "created_at": created_at,
            "status": status
        })

        # Stampa in modo leggibile
        status_str = " | ".join(f"{k}:{'✔' if v else '✘'}" for k, v in status.items())
        print(f"{created_at}  {MEname}")
        print(f"  ID: {job_id}")
        print(f"  Stato: {status_str}")
        print("-" * 80)

    return jobs
    
def delete_empty_jobs(base_path="outputs"):
    """
    Cancella tutti i job che hanno S0 = NULL dal database e rimuove le directory di output.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Prendo tutti i job con S0 vuota
    c.execute(f"SELECT job_id, MEname FROM {TABLE_NAME} WHERE S0 IS NULL")
    jobs_to_delete = c.fetchall()

    if not jobs_to_delete:
        print("Nessun job con S0 vuota trovato.")
        conn.close()
        return

    print(f"Trovati {len(jobs_to_delete)} job con S0 vuota. Li cancello...")
    for job_id, MEname in jobs_to_delete:
        job_dir = os.path.join(base_path, f"{MEname}-{job_id}")
        
        # Cancella directory se esiste
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
            print(f"  🗑  Directory rimossa: {job_dir}")
        else:
            print(f"  ⚠️  Directory non trovata: {job_dir}")

        # Rimuovi entry dal DB
        c.execute(f"DELETE FROM {TABLE_NAME} WHERE job_id = ?", (job_id,))
        print(f"  🗑  Entry DB rimossa: job_id={job_id}")

    conn.commit()
    conn.close()
    print("Operazione completata.")
