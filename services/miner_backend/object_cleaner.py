import time
import os
from minio import Minio
import structlog
from datetime import datetime, timezone

logger = structlog.get_logger()

class MinioCleanup:
    def __init__(self):
        self.bucket_name = os.getenv("MINIO_BUCKET", "condense_miner")
        self.minio_client = Minio(
            f"localhost:{os.getenv('MINIO_PORT')}",
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=False,
        )
        
    def cleanup_old_objects(self, max_age_hours=1):
        """Delete objects older than max_age_hours"""
        try:
            # Get current time in UTC
            now = datetime.now(timezone.utc)
            
            # List all objects in the bucket
            objects = list(self.minio_client.list_objects(self.bucket_name))
            total_files = len(objects)
            expired_files = 0
            
            for obj in objects:
                # Calculate age in hours
                age = (now - obj.last_modified).total_seconds() / 3600
                
                if age > max_age_hours:
                    self.minio_client.remove_object(self.bucket_name, obj.object_name)
                    expired_files += 1
                    logger.info(
                        "deleted_object",
                        object_name=obj.object_name,
                        age_hours=round(age, 2)
                    )
            
            # Log summary statistics
            logger.info(
                "cleanup_summary",
                total_files=total_files,
                expired_files=expired_files,
                remaining_files=total_files - expired_files,
                bucket_name=self.bucket_name
            )
                    
        except Exception as e:
            logger.exception("cleanup_failed", error=str(e))

def main():
    cleanup = MinioCleanup()
    
    # Run cleanup every 5 minutes
    while True:
        cleanup.cleanup_old_objects()
        time.sleep(300)  # Sleep for 5 minutes

if __name__ == "__main__":
    main() 