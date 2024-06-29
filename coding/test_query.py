import torch
from query import QueryModule

#Tests
def test_queryModule():
    qm = QueryModule()

    iree_qm = torch.compile(qm, backend="turbine_cpu")
    tables = []
    for _ in range (0,3):
        tables.append(torch.rand((10,10), requires_grad=False, dtype=torch.float)*100)
    tableDict = dict([("A",tables[0]), ("B",tables[1]),("C",tables[2])])
    bColumnNamesDict = dict([("X",0), ("Y",1), ("Z",2)])
    print("Full table B:")
    print(qm.from_table(tableDict, "B"))
    print("Table B with rows 2 and 7:")
    print(qm.select_rows(qm.from_table(tableDict, "B"), [2,7]))
    print("Table B with columns 0 and 1:")
    print(qm.select_column(qm.from_table(tableDict, "B"),[0,1]))
    print("Table B with rows that have a value greater than 0.5 in X column:")
    print(qm.filter_rows(qm.from_table(tableDict, "B"), bColumnNamesDict, [">", "X", 50.0]))
    print("Table B with rows that have a value equal to 0.5 in the Z column:")
    print(qm.filter_rows(qm.from_table(tableDict, "B"), bColumnNamesDict, ["=", "Z", 50.0]))
    print("Table B with rows that have a value less than 0.5 in the Y column:")
    print(qm.filter_rows(qm.from_table(tableDict, "B"), bColumnNamesDict, ["<", "Y", 50.0]))
    print("Table B with the columns X and Y:")
    print(qm.filter_column(qm.from_table(tableDict, "B"), bColumnNamesDict, ["X", "Y"]))

    print("SELECT X, Y FROM B WHERE Z < 70:")
    print(qm.filter_column(qm.filter_rows(qm.from_table(tableDict, "B"), bColumnNamesDict, ["<", "Z", 70]), bColumnNamesDict, ["X", "Y"]))

    print(qm.process_query("Select X, Y From B Where Z < 70"))