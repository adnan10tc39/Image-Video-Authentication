import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiosqlite

JobProcessor = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass
class JobRecord:
    id: str
    type: str
    payload: Dict[str, Any]
    status: str
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    created_at: str
    updated_at: str


class JobManager:
    def __init__(
        self,
        db_path: Path,
        worker_count: int = 1,
    ) -> None:
        self.db_path = str(db_path)
        self.worker_count = worker_count
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._processors: Dict[str, JobProcessor] = {}
        self._workers: List[asyncio.Task[None]] = []
        self._startup_lock = asyncio.Lock()
        self._started = False

    def register_processor(self, job_type: str, processor: JobProcessor) -> None:
        self._processors[job_type] = processor

    async def start(self) -> None:
        async with self._startup_lock:
            if self._started:
                return

            await self._init_db()
            await self._requeue_incomplete_jobs()

            for _ in range(self.worker_count):
                task = asyncio.create_task(self._worker_loop())
                self._workers.append(task)

            self._started = True

    async def stop(self) -> None:
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

    async def enqueue_job(self, job_type: str, payload: Dict[str, Any]) -> str:
        if job_type not in self._processors:
            raise ValueError(f"No processor registered for job type '{job_type}'")

        job_id = uuid.uuid4().hex
        now = self._now()
        payload_json = json.dumps(payload, ensure_ascii=False)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO jobs (id, type, payload, status, result, error, created_at, updated_at)
                VALUES (?, ?, ?, 'queued', NULL, NULL, ?, ?)
                """,
                (job_id, job_type, payload_json, now, now),
            )
            await db.commit()

        await self.queue.put(job_id)
        return job_id

    async def fetch_job(self, job_id: str) -> Optional[JobRecord]:
        row = await self._fetch_row(job_id)
        if not row:
            return None
        return self._row_to_record(row)

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                row = await self._fetch_row(job_id)
                if not row:
                    continue

                record = self._row_to_record(row)
                if record.status not in {"queued", "running"}:
                    continue

                await self._update_status(job_id, "running")

                processor = self._processors.get(record.type)
                if not processor:
                    await self._update_status(
                        job_id,
                        "failed",
                        error=f"No processor registered for job type '{record.type}'",
                    )
                    continue

                try:
                    result = await processor(record.payload)
                    await self._update_status(job_id, "completed", result=result)
                except Exception as exc:  # pylint: disable=broad-except
                    await self._update_status(job_id, "failed", error=str(exc))
            finally:
                self.queue.task_done()

    async def _update_status(
        self,
        job_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        result_json = json.dumps(result, ensure_ascii=False) if result is not None else None
        now = self._now()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE jobs
                SET status = ?, result = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, result_json, error, now, job_id),
            )
            await db.commit()

    async def _fetch_row(self, job_id: str) -> Optional[aiosqlite.Row]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = await cursor.fetchone()
            await cursor.close()
        return row

    async def _init_db(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await db.commit()

    async def _requeue_incomplete_jobs(self) -> None:
        now = self._now()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE jobs
                SET status = 'queued', updated_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )
            await db.commit()
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT id FROM jobs
                WHERE status = 'queued'
                """
            )
            rows = await cursor.fetchall()
            await cursor.close()

        for row in rows or []:
            await self.queue.put(row["id"])

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> JobRecord:
        payload = json.loads(row["payload"]) if row["payload"] else {}
        result = json.loads(row["result"]) if row["result"] else None
        return JobRecord(
            id=row["id"],
            type=row["type"],
            payload=payload,
            status=row["status"],
            result=result,
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

