import torch
import operator


ops = {">":operator.gt, ">=":operator.ge, "<":operator.lt, "<=":operator.le, "=":operator.eq, "<>":operator.ne}


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
    #TODO: multiple condition statements
    def filter_rows(self, table, column_name_dict, filter_cond, filter_value, filter_column_name):
        indices = self.index_by_name(column_name_dict, filter_column_name)
        return table[ops[filter_cond](table[:,indices[0]], filter_value)]

    #returns a table with only the columns that are given by name
    def filter_column(self, table, column_name_dict, column_names_filter):
        return self.select_column(table, self.index_by_name(column_name_dict, column_names_filter))


#Tests
qm = QueryModule()

iree_qm = torch.compile(qm, backend="turbine_cpu")
tables = []
for _ in range (0,3):
    tables.append(torch.rand((10,10), requires_grad=False, dtype=torch.float))
tableDict = dict([("A",tables[0]), ("B",tables[1]),("C",tables[2])])
bColumnNamesDict = dict([("X",0), ("Y",1), ("Z",2)])
print("Full table B:")
print(qm.from_table(tableDict, "B"))
print("Table B with rows 2 and 7:")
print(qm.select_rows(qm.from_table(tableDict, "B"), [2,7]))
print("Table B with columns 0 and 1:")
print(qm.select_column(qm.from_table(tableDict, "B"),[0,1]))
print("Table B with rows that have a value greater than 0.5 in X column:")
print(qm.filter_rows(qm.from_table(tableDict, "B"), bColumnNamesDict, ">", 0.5, "X"))
print("Table B with rows that have a value equal to 0.5 in the Z column:")
print(qm.filter_rows(qm.from_table(tableDict, "B"), bColumnNamesDict, "=", 0.5, "Z"))
print("Table B with rows that have a value less than 0.5 in the Y column:")
print(qm.filter_rows(qm.from_table(tableDict, "B"), bColumnNamesDict, "<", 0.5, "Y"))
print("Table B with the columns X and Y:")
print(qm.filter_column(qm.from_table(tableDict, "B"), bColumnNamesDict, ["X", "Y"]))

print("SELECT X, Y FROM B WHERE Z < 0.7:")
print(qm.filter_column(qm.filter_rows(qm.from_table(tableDict, "B"), bColumnNamesDict, "<", 0.7, "Z" ), bColumnNamesDict, ["X", "Y"]))