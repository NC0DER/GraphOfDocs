from neo4j import GraphDatabase, CypherError, ServiceUnavailable

class Neo4jDatabase(object): 
    """
    Wrapper class to handle the database 
    more efficiently, by abstracting repeating code.
    """
    def __init__(self, uri, user, password): # Open the database and authenticate.
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def execute(self, query, mode): # Execute queries in the database.
        with self._driver.session() as session:
            if (mode == 'r'): # Reading query.
                result = session.read_transaction(self.__execute, query)
            elif(mode == 'w'): # Writing query.
                result = session.write_transaction(self.__execute, query)
            else:
                raise TypeError('Execution mode can either be (r)ead or (w)rite!')
            return result

    @staticmethod # private method.
    def __execute(tx, query):
        result = tx.run(query)
        try:
            return result.values() # Return node, relationship values in a list of tuples.
        except CypherError as err: pass # Handle the erroneous query instead of breaking the execution.
