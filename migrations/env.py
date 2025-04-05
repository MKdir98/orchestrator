from logging.config import fileConfig
import os
import sys

# اضافه کردن مسیر پروژه به Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# این خط را حتماً قبل از import مدل‌ها قرار دهید
from orchestrator.models.base import Base

# تنظیمات Alembic
config = context.config

# تنظیمات لاگینگ
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# این بخش حیاتی است - باید به درستی تنظیم شود
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()