from sqlalchemy import text
from database import Base, engine, SessionLocal
from models import Space, Cohort  # CohortUser is created via metadata, no direct use here


def run_migration():
    try:
        # 1. Create any NEW tables from updated models (cohorts, cohort_users, etc.)
        Base.metadata.create_all(bind=engine)
        print("Base.metadata.create_all() executed (new tables created if missing).")

        # 2. Ensure `cohort_id` column exists on `spaces` table
        # NOTE: SQLAlchemy's create_all will NOT add columns to existing tables.
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    ALTER TABLE spaces
                    ADD COLUMN IF NOT EXISTS cohort_id INTEGER
                    """
                )
            )
        print("Ensured spaces.cohort_id column exists.")

        # 3. Create a default cohort for legacy spaces (if not already present)
        db = SessionLocal()
        default_code = "LEGACY"
        default_name = "Legacy Cohort"

        default_cohort = (
            db.query(Cohort)
            .filter(Cohort.code == default_code)
            .first()
        )

        if not default_cohort:
            default_cohort = Cohort(
                name=default_name,
                code=default_code,
                is_active=True,
            )
            db.add(default_cohort)
            db.commit()
            db.refresh(default_cohort)
            print(f"Created default cohort: {default_name} ({default_code})")
        else:
            print(f"Default cohort already exists: {default_cohort.name} ({default_cohort.code})")

        # 4. Assign all existing spaces that have NULL cohort_id to the default cohort
        updated_rows = (
            db.query(Space)
            .filter(Space.cohort_id.is_(None))
            .update(
                {Space.cohort_id: default_cohort.id},
                synchronize_session=False,
            )
        )
        db.commit()
        print(f"Updated {updated_rows} existing spaces to default cohort_id={default_cohort.id}.")

        # 5. (Optional, stricter) Enforce NOT NULL on cohort_id now that everything is populated.
        # Comment out if you prefer to keep it nullable at DB level.
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    ALTER TABLE spaces
                    ALTER COLUMN cohort_id SET NOT NULL
                    """
                )
            )
        print("Set spaces.cohort_id to NOT NULL.")

        # 6. (Optional) Add a foreign key constraint from spaces.cohort_id → cohorts.id
        # This may fail if the constraint already exists; we ignore that error.
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        ALTER TABLE spaces
                        ADD CONSTRAINT spaces_cohort_id_fkey
                        FOREIGN KEY (cohort_id) REFERENCES cohorts(id)
                        """
                    )
                )
            print("Ensured foreign key spaces.cohort_id → cohorts.id exists.")
        except Exception as fk_err:
            # Some Postgres versions do not support IF NOT EXISTS on constraints,
            # so it's fine if this fails after first run.
            print(f"Warning: could not create FK constraint (may already exist): {fk_err}")

        # 7. Run a quick test query
        spaces = db.query(Space).all()
        print("Spaces after migration:", spaces)

        db.close()
        print("Migration completed successfully!")

    except Exception as e:
        print("Migration failed:", e)


if __name__ == "__main__":
    run_migration()
