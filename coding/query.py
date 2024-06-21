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