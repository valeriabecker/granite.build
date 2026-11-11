#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Move registered-artifact uuids out of ``build_id`` in the lineage tables (#583).

Before #583 a registration's lineage record stored the artifact uuid in
``build_id``. New records store it under ``artifact_id``. This one-off,
idempotent pass rewrites the old records in ``gb_lineage_job`` and their rows in
the lineage index. Counting tolerates unmigrated records, so it can run at any
time; W&B runs are not migrated (see ``WandBLineageStore.count_release_ids``).
"""

import click

from gbserver.lineage.db_jobstats import migrate_registration_build_ids
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.types.context import CliEnvironment, pass_environment


@click.command()
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Count the records that would be migrated and exit without writing.",
)
@pass_environment
def cli(ctx: CliEnvironment, dry_run: bool) -> None:
    """Move registration artifact uuids from build_id to artifact_id."""
    storage = get_admin_storage()
    count = migrate_registration_build_ids(
        storage.lineage_job_storage,
        row_storage=storage.lineage_row_storage,
        dry_run=dry_run,
    )
    verb = "would be migrated" if dry_run else "migrated"
    click.echo(f"{count} registration lineage record(s) and index row(s) {verb}.")
