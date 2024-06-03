class QueryModule():
    #Converts the SQL query into a parsed query tree
    def SQLquery_Into_Tree(String query):
        output parse_tree
    #Processes the parsed query tree and returns the output of the query
    def Process_Parse_Tree(Tree parse_tree):
        output query_output
    #Provides the table with the specified columns   
    def Select_From(Dictionary tables, String table_name, String column_names):
        output Tensor table_with_specified_columns
    #Provides the table with the entrys, that fullfill the given conditions
    def Select_Where(Tensor table, String conditions):
        output Tensor table_with_conditions_fulfilled
    #Provides the aggregated table/values
    def Aggregate(Tensor table, String aggregate):
        output Tensor aggregate