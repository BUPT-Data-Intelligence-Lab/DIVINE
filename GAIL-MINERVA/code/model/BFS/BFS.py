from queue import Queue
import random


# use queue to achieve BFS algorithm
def BFS(kb, entity1, entity2):
    res = foundPaths(kb)
    res.markFound(entity1, None, None)
    q = Queue()
    q.put(entity1)
    while (not q.empty()):
        curNode = q.get()
        for path in kb.getPathsFrom(curNode):
            nextEntity = path.connected_entity
            connectRelation = path.relation
            # if nextEntity has never been found, put it into queue
            if (not res.isFound(nextEntity)):
                q.put(nextEntity)
                res.markFound(nextEntity, curNode, connectRelation)
            # BFS search is done
            if (nextEntity == entity2):
                entity_list, path_list = res.reconstructPath(entity1, entity2)
                return (True, entity_list, path_list)
    return (False, None, None)


def test():
    pass


class foundPaths(object):
    def __init__(self, kb):
        self.entities = {}
        for entity, relations in kb.entities.items():
            self.entities[entity] = (False, "", "")

    def isFound(self, entity):
        return self.entities[entity][0]

    def markFound(self, entity, prevNode, relation):
        self.entities[entity] = (True, prevNode, relation)

    def reconstructPath(self, entity1, entity2):
        entity_list = []
        path_list = []
        curNode = entity2
        while (curNode != entity1):
            entity_list.append(curNode)

            path_list.append(self.entities[curNode][2])
            curNode = self.entities[curNode][1]
        entity_list.append(curNode)
        # because we record the paths in this function from en2 to en1, so we need to reverse the list.
        entity_list.reverse()
        path_list.reverse()
        return (entity_list, path_list)

    def __str__(self):
        res = ""
        for entity, status in self.entities.items():
            res += entity + "[{},{},{}]".format(status[0], status[1], status[2])
        return res
