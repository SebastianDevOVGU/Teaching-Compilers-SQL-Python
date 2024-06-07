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
    #returns a table with only the rows that match the condition statement
    def filter_rows(self, table, filter_cond, filter_value, filter_column):
        return table[torch.nonzero(ops[filter_cond](self.select_column(table, filter_column), filter_value))]


#Tests
qm = QueryModule()

iree_qm = torch.compile(qm, backend="turbine_cpu")
tables = []
for _ in range (0,3):
    tables.append(torch.rand((10,10), requires_grad=False, dtype=torch.float))
tableDict = dict([("A",tables[0]), ("B",tables[1]),("C",tables[2])])
print("Full table B:")
print(qm.from_table(tableDict, "B"))
print("Table B with rows 2 and 7:")
print(qm.select_rows(qm.from_table(tableDict, "B"), [2,7]))
print("Table B with columns 0 and 1:")
print(qm.select_column(qm.from_table(tableDict, "B"),[0,1]))
print("Table B with rows that have a value greater than 0.5 in the first column:")
print(qm.filter_rows(qm.from_table(tableDict, "B"), ">", 0.5, [0]))
print("Table B with rows that have a value equal to 0.5 in the third column:")
print(qm.filter_rows(qm.from_table(tableDict, "B"), "=", 0.5, [2]))
print("Table B with rows that have a value less than 0.5 in the second column:")
print(qm.filter_rows(qm.from_table(tableDict, "B"), "<", 0.5, [1]))