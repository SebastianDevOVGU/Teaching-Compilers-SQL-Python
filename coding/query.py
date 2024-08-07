import torch
import operator
from pglast import parser, enums, ast


ops = {">":operator.gt, ">=":operator.ge, "<":operator.lt, "<=":operator.le, "=":operator.eq, "<>":operator.ne}

def split_SelFroWhe(query):
    root = parser.parse_sql(query)
    stmt = root[0].stmt
    selectstmt, fromstmt, wherestmt = [], [], []
    for t in stmt.targetList:
        selectstmt.append(t.val.fields[0].sval)
    for f in stmt.fromClause:
        fromstmt.append(f.relname)
    if not isinstance(stmt.whereClause, ast.A_Expr):
        if stmt.whereClause.boolop is enums.BoolExprType.AND_EXPR:
            wherestmt.append("AND")
        elif stmt.whereClause.boolop is enums.BoolExprType.OR_EXPR:
            wherestmt.append("OR")
        elif stmt.whereClause.boolop is enums.BoolExprType.NOT_EXPR:
            wherestmt.append("NOT")
        else:
            return None
        for w in stmt.whereClause.args:
            rexpr = None
            if isinstance(w.rexpr , ast.A_Const):
                if isinstance(w.rexpr.val, ast.Integer):
                    rexpr = w.rexpr.val.ival
                elif isinstance(w.rexpr.val, ast.Float):
                    rexpr = w.rexpr.val.fval    
            else:
                rexpr = w.rexpr.fields[0].sval
            wherestmt.append([w.name[0].sval, w.lexpr.fields[0].sval, rexpr])
    else:
        w = stmt.whereClause
        rexpr = None
        if isinstance(w.rexpr , ast.A_Const):
            if isinstance(w.rexpr.val, ast.Integer):
                rexpr = w.rexpr.val.ival
            elif isinstance(w.rexpr.val, ast.Float):
                rexpr = w.rexpr.val.fval    
        else:
            rexpr = w.rexpr.fields[0].sval
        wherestmt.append([w.name[0].sval, w.lexpr.fields[0].sval, rexpr])
    return selectstmt, fromstmt, wherestmt

class QueryModule(torch.nn.Module):
    def __init__(self):
        super(QueryModule, self).__init__()

    #returns the table specified by its name
    def from_table(self, tableDict, tableSelected):
        return tableDict[tableSelected]

    #returns a table with only the specified rows (by index)
    def select_rows(self, table, rows):
        indices = torch.tensor(rows)
        return torch.index_select(table, 0, indices)

    #returns a table with only the specified columns (by index)
    def select_column(self, table, columns):
        indices = torch.tensor(columns)
        return torch.index_select(table, 1, indices)
    #helper function for getting the indices of the columns
    def index_by_name(self, dict, names):
        indices = []
        for name in names:
            indices.append(dict[name])
        return indices
    
    #returns a table with only the rows that match the condition statement
    def filter_rows(self, table, column_name_dict, condition):
        filter_column_names = []
        for expr in condition[1:]:
            filter_column_names.append(expr[1])
            index = self.index_by_name(column_name_dict, filter_column_names)
            table = table[ops[expr[0]](table[:,index[0]], expr[2])]
        return table

    #returns a table with only the columns that are given by name
    def filter_column(self, table, column_name_dict, column_names_filter):
        return self.select_column(table, self.index_by_name(column_name_dict, column_names_filter))
    
    #process the query
    def process_query(self, tableDict, columnNamesDict, query):
        targets, table_names, condition = split_SelFroWhe(query)
        tables = []
        for t_name in table_names:
            tables.append(self.filter_column(self.filter_rows(self.from_table(tableDict, t_name), columnNamesDict, condition), columnNamesDict, targets))
        return tables