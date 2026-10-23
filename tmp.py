from utils.jt1993_data_download import get_wrds_connection

monitor = get_wrds_connection(username="philxu27")

activity_sql = """
select pid, state,
       case when state = 'active'
            then clock_timestamp() - query_start
       end as running_for,
       wait_event_type, wait_event,
       left(query, 500) as sql
from pg_stat_activity
where usename = current_user
  and pid <> pg_backend_pid()
order by query_start
"""

print(monitor.raw_sql(activity_sql).to_string(index=False))