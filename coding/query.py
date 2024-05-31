import torch

class QueryModule(torch.nn.Module):
    def __init__(self):
        super(QueryModule, self).__init__()
    #returns the table specified by its name
    def from_table(self, tables, tableDict, tableSelected):
        return tables[tableDict[tableSelected]]
    #returns a table with only the specified rows (by index)
    def select_rows(self, table, rows):
        indices = torch.tensor(rows)
        return torch.index_select(table, 0, indices)
    #returns a table with only the specified columns (by index)
    def select_column(self, table, columns):
        indices = torch.tensor(columns)
        return torch.index_select(table, 1, indices)


#Tests
qm = QueryModule()

tableDict = dict([("A",0), ("B",1),("C",2)])
tables = []
for _ in range (0,3):
    tables.append(torch.rand((10,10), requires_grad=False, dtype=torch.float))
print(qm.from_table(tables, tableDict, "B"))
print(qm.select_rows(qm.from_table(tables, tableDict, "B"), [2,7]))
print(qm.select_column(qm.from_table(tables, tableDict, "B"),[0,1]))