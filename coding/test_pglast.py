from query import split_SelFroWhe


def test_pglast_to_list():

    print(split_SelFroWhe('SELECT name, age, id FROM students, course WHERE age > 25 and course = DB2 and grade=1.3'))

