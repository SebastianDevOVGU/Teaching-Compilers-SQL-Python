import torch
from query import QueryModule

#Tests
def test_queryModule():
    qm = QueryModule()

    iree_qm = torch.compile(qm, backend="turbine_cpu")
    tables = []
    for _ in range (0,3):
        tables.append(torch.rand((10,10), requires_grad=False, dtype=torch.float)*100)
    tableDict = dict([("A",tables[0]), ("b",tables[1]),("C",tables[2])])
    bColumnNamesDict = dict([("x",0), ("y",1), ("z",2)])
    print("Full table B:")
    print(qm.from_table(tableDict, "b"))
    print("Table B with rows 2 and 7:")
    print(qm.select_rows(qm.from_table(tableDict, "b"), [2,7]))
    print("Table B with columns 0 and 1:")
    print(qm.select_column(qm.from_table(tableDict, "b"),[0,1]))
    print("Table B with rows that have a value greater than 0.5 in x column:")
    print(qm.filter_rows(qm.from_table(tableDict, "b"), bColumnNamesDict, ["AND",[">", "x", 50.0]]))
    print("Table B with rows that have a value equal to 0.5 in the z column:")
    print(qm.filter_rows(qm.from_table(tableDict, "b"), bColumnNamesDict, ["AND",["=", "z", 50.0]]))
    print("Table B with rows that have a value less than 0.5 in the y column:")
    print(qm.filter_rows(qm.from_table(tableDict, "b"), bColumnNamesDict, ["AND",["<", "y", 50.0]]))
    print("Table B with the columns x and y:")
    print(qm.filter_column(qm.from_table(tableDict, "b"), bColumnNamesDict, ["x", "y"]))

    print("SELECT x, y FROM B WHERE z < 70:")
    print(qm.filter_column(qm.filter_rows(qm.from_table(tableDict, "b"), bColumnNamesDict, ["AND",["<", "z", 70]]), bColumnNamesDict, ["x", "y"]))

    print(qm.process_query(tableDict, bColumnNamesDict, "Select x, y From B Where z < 70 AND z > 10"))

test_queryModule()