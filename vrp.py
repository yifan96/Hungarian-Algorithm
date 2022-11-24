"""Simple Vehicles Routing Problem (VRP).

   This is a sample using the routing library python wrapper to solve a VRP
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters.
"""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class VRP_solver(object):
    def __init__(self, cost_matrix, num_agents, depots):
        # self.cost_matrix = cost_matrix
        # self.num_agents = num_agents
        # self.depots = depots
        self.data = {}
        self.data['distance_matrix'] = cost_matrix
        self.data['num_vehicles'] = num_agents
        self.data['depots'] = depots
        self.data['ends'] = depots

    def print_solution(self, manager, routing, solution):
        """Prints solution on console."""
        #print(f'Objective: {solution.ObjectiveValue()}')
        max_route_distance = 0
        total_distance = 0
        for vehicle_id in range(self.data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            while not routing.IsEnd(index):
                plan_output += ' {} -> '.format(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            plan_output += '{}\n'.format(manager.IndexToNode(index))
            plan_output += 'Distance of the route: {}m\n'.format(route_distance)
            #print(plan_output)
            max_route_distance = max(route_distance, max_route_distance)
            total_distance = total_distance + route_distance
            #print("route_distance", total_distance)
        #print('Maximum of the route distances: {}m'.format(max_route_distance))


        assignment_result = [[] for i in range(self.data['num_vehicles'])]
        for vehicle_id in range(self.data['num_vehicles']):
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                assignment_result[vehicle_id].append(manager.IndexToNode(index))
        # delete last element, not returning to initial position
        #print(assignment_result)
        for x in assignment_result:
            x.pop()
        assignment_result = [[(x - self.data['num_vehicles']) for x in row] for row in assignment_result]

        return assignment_result, max_route_distance, total_distance



    def solve(self):
        """Entry point of the program."""
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(self.data['distance_matrix']),
                                               self.data['num_vehicles'], self.data['depots'], self.data['ends'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            3000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)

        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(1)

        # # Setting first solution heuristic.
        # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        # search_parameters.first_solution_strategy = (
        #     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
        # # search_parameters.time_limit.seconds = 30
        search_parameters.solution_limit = 500
        search_parameters.log_search = False

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            assingment_result, max_route_distance, total_distance = self.print_solution(manager, routing, solution)
        else:
            print('No solution found !')
        return assingment_result, max_route_distance, total_distance






