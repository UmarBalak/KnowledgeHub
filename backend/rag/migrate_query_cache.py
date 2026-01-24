from database import Base, engine, SessionLocal
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_query_logs():
    """
    Migration: Update query_logs table for privacy-conscious caching
    - Removes query_text (privacy concern)
    - Adds query_hash for exact matching
    - Adds query_embedding for semantic matching
    - Adds space_id for filtering
    - Adds hit_count for analytics
    """
    db = SessionLocal()
    
    try:
        logger.info("🔄 Starting query_logs migration...")
        
        migrations = [
            {
                "name": "Drop query_text column",
                "sql": "ALTER TABLE query_logs DROP COLUMN IF EXISTS query_text;"
            },
            {
                "name": "Add query_hash column",
                "sql": "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS query_hash VARCHAR(64);"
            },
            {
                "name": "Add query_embedding column",
                "sql": "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS query_embedding JSON;"
            },
            {
                "name": "Add space_id column",
                "sql": "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS space_id INTEGER;"
            },
            {
                "name": "Add hit_count column",
                "sql": "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS hit_count INTEGER DEFAULT 1;"
            },
            {
                "name": "Create index on query_hash",
                "sql": "CREATE INDEX IF NOT EXISTS idx_query_hash ON query_logs(query_hash);"
            },
            {
                "name": "Create index on space_id",
                "sql": "CREATE INDEX IF NOT EXISTS idx_space_id ON query_logs(space_id);"
            },
            {
                "name": "Create composite index",
                "sql": "CREATE INDEX IF NOT EXISTS idx_query_hash_space ON query_logs(query_hash, space_id);"
            },
            {
                "name": "Set default hit_count",
                "sql": "UPDATE query_logs SET hit_count = 1 WHERE hit_count IS NULL;"
            }
        ]
        
        for i, migration in enumerate(migrations, 1):
            try:
                logger.info(f"  [{i}/{len(migrations)}] {migration['name']}...")
                db.execute(text(migration['sql']))
                db.commit()
                logger.info(f"      ✓ Success")
            except Exception as e:
                logger.warning(f"      ⚠️  {e}")
                db.rollback()
        
        logger.info("✅ Migration completed successfully!")
        
        # Verify changes
        result = db.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'query_logs'
            ORDER BY ordinal_position;
        """))
        
        logger.info("\n📋 Current query_logs schema:")
        for row in result:
            logger.info(f"   - {row[0]}: {row[1]}")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    migrate_query_logs()
