from pglast import parse_sql, ast
root = parse_sql('SELECT name FROM students WHERE age > 25')
rawstmt = root[0]
fromstmt = root[0]
stmt = rawstmt.stmt
target = stmt.targetList[0]
from_clause = stmt.fromClause[0]
where_clause = stmt.whereClause
const_value = target.val.fields
print("The select value is: ", const_value[0].sval)
print("The from value is: ",from_clause.relname)
print("The where clause is: ",where_clause.lexpr.fields[0].sval,"",where_clause.name[0].sval,"",where_clause.rexpr.val.ival)