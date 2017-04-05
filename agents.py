
#from wumpus_agent import *
from time import clock
from utils import *
import random, copy
import itertools, re


class Thing(object):
    """This represents any physical object that can appear in an Environment.
    You subclass Thing to get the things you want.  Each thing can have a
    .__name__  slot (used for output only)."""

    def __repr__(self):
        return '<%s>' % getattr(self, '__name__', self.__class__.__name__)

    def is_alive(self):
        """Things that are 'alive' should return true."""
        return hasattr(self, 'alive') and self.alive

    def show_state(self):
        """Display the agent's internal state.  Subclasses should override."""
        print "I don't know how to show_state."

    def display(self, canvas, x, y, width, height):
        """Display an image of this Thing on the canvas."""
        pass


class Agent(Thing):
    """An Agent is a subclass of Thing with one required slot,
    .program, which should hold a function that takes one argument, the
    percept, and returns an action. (What counts as a percept or action
    will depend on the specific environment in which the agent exists.)
    Note that 'program' is a slot, not a method.  If it were a method,
    then the program could 'cheat' and look at aspects of the agent.
    It's not supposed to do that: the program can only look at the
    percepts.  An agent program that needs a model of the world (and of
    the agent itself) will have to build and maintain its own model.
    There is an optional slot, .performance, which is a number giving
    the performance measure of the agent in its environment."""

    def __init__(self, program = None):
        self.alive = True
        self.bump = False
        if program is None:

            def program(percept):
                return raw_input('Percept=%s; action? ' % percept)

        assert callable(program)
        self.program = program

    def can_grab(self, thing):
        """Returns True if this agent can grab this thing.
        Override for appropriate subclasses of Agent and Thing."""
        return False


def TraceAgent(agent):
    """Wrap the agent's program to print its input and output. This will let
    you see what the agent is doing in the environment."""
    old_program = agent.program

    def new_program(percept):
        action = old_program(percept)
        print '%s perceives %s and does %s' % (agent, percept, action)
        return action

    agent.program = new_program
    return agent


def TableDrivenAgentProgram(table):
    """This agent selects an action based on the percept sequence.
    It is practical only for tiny domains.
    To customize it, provide as table a dictionary of all
    {percept_sequence:action} pairs. [Fig. 2.7]"""
    percepts = []

    def program(percept):
        percepts.append(percept)
        action = table.get(tuple(percepts))
        return action

    return program


def RandomAgentProgram(actions):
    """An agent that chooses an action at random, ignoring all percepts."""
    return lambda percept: random.choice(actions)


def SimpleReflexAgentProgram(rules, interpret_input):
    """This agent takes action based solely on the percept. [Fig. 2.10]"""

    def program(percept):
        state = interpret_input(percept)
        rule = rule_match(state, rules)
        action = rule.action
        return action

    return program


def ModelBasedReflexAgentProgram(rules, update_state):
    """This agent takes action based on the percept and state. [Fig. 2.12]"""

    def program(percept):
        program.state = update_state(program.state, program.action, percept)
        rule = rule_match(program.state, rules)
        action = rule.action
        return action

    program.state = program.action = None
    return program


def rule_match(state, rules):
    """Find the first rule that matches state."""
    for rule in rules:
        if rule.matches(state):
            return rule


loc_A, loc_B = (0, 0), (1, 0)

def RandomVacuumAgent():
    """Randomly choose one of the actions from the vacuum environment."""
    return Agent(RandomAgentProgram(['Right',
     'Left',
     'Suck',
     'NoOp']))


def TableDrivenVacuumAgent():
    """[Fig. 2.3]"""
    table = {((loc_A, 'Clean'),): 'Right',
     ((loc_A, 'Dirty'),): 'Suck',
     ((loc_B, 'Clean'),): 'Left',
     ((loc_B, 'Dirty'),): 'Suck',
     ((loc_A, 'Clean'), (loc_A, 'Clean')): 'Right',
     ((loc_A, 'Clean'), (loc_A, 'Dirty')): 'Suck',
     ((loc_A, 'Clean'), (loc_A, 'Clean'), (loc_A, 'Clean')): 'Right',
     ((loc_A, 'Clean'), (loc_A, 'Clean'), (loc_A, 'Dirty')): 'Suck'}
    return Agent(TableDrivenAgentProgram(table))


def ReflexVacuumAgent():
    """A reflex agent for the two-state vacuum environment. [Fig. 2.8]"""

    def program((location, status)):
        if status == 'Dirty':
            return 'Suck'
        if location == loc_A:
            return 'Right'
        if location == loc_B:
            return 'Left'

    return Agent(program)


def ModelBasedVacuumAgent():
    """An agent that keeps track of what locations are clean or dirty."""
    model = {loc_A: None,
     loc_B: None}

    def program((location, status)):
        """Same as ReflexVacuumAgent, except if everything is clean, do NoOp."""
        model[location] = status
        if model[loc_A] == model[loc_B] == 'Clean':
            return 'NoOp'
        if status == 'Dirty':
            return 'Suck'
        if location == loc_A:
            return 'Right'
        if location == loc_B:
            return 'Left'

    return Agent(program)


class Environment(object):
    """Abstract class representing an Environment.  'Real' Environment classes
    inherit from this. Your Environment will typically need to implement:
        percept:           Define the percept that an agent sees.
        execute_action:    Define the effects of executing an action.
                           Also update the agent.performance slot.
    The environment keeps a list of .things and .agents (which is a subset
    of .things). Each agent has a .performance slot, initialized to 0.
    Each thing has a .location slot, even though some environments may not
    need this."""

    def __init__(self):
        self.things = []
        self.agents = []

    def thing_classes(self):
        return []

    def percept(self, agent):
        """Return the percept that the agent sees at this point. (Implement this.)"""
        abstract

    def execute_action(self, agent, action):
        """Change the world to reflect this action. (Implement this.)"""
        abstract

    def default_location(self, thing):
        """Default location to place a new thing with unspecified location."""
        return None

    def exogenous_change(self):
        """If there is spontaneous change in the world, override this."""
        pass

    def is_done(self):
        """By default, we're done when we can't find a live agent."""
        return not any((agent.is_alive() for agent in self.agents))

    def step(self):
        """Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do.  If there are interactions between them, you'll need to
        override this method."""
        if not self.is_done():
            actions = [ agent.program(self.percept(agent)) for agent in self.agents ]
            for agent, action in zip(self.agents, actions):
                self.execute_action(agent, action)

            self.exogenous_change()

    def run(self, steps = 1000):
        """Run the Environment for given number of time steps."""
        for step in range(steps):
            if self.is_done():
                return
            self.step()

    def list_things_at(self, location, tclass = Thing):
        """Return all things exactly at a given location."""
        return [ thing
                 for thing in self.things
                 if thing.location == location and isinstance(thing, tclass) ]

    def some_things_at(self, location, tclass = Thing):
        """Return true if at least one of the things at location
        is an instance of class tclass (or a subclass)."""
        return self.list_things_at(location, tclass) != []

    def add_thing(self, thing, location = None):
        """Add a thing to the environment, setting its location. For
        convenience, if thing is an agent program we make a new agent
        for it. (Shouldn't need to override this."""
        if not isinstance(thing, Thing):
            thing = Agent(thing)
        assert thing not in self.things, "Don't add the same thing twice"
        thing.location = location or self.default_location(thing)
        self.things.append(thing)
        if isinstance(thing, Agent):
            thing.performance = 0
            self.agents.append(thing)

    def delete_thing(self, thing):
        """Remove a thing from the environment."""
        try:
            self.things.remove(thing)
        except ValueError as e:
            print e
            print '  in Environment delete_thing'
            print '  Thing to be removed: %s at %s' % (thing, thing.location)
            print '  from list: %s' % [ (thing, thing.location) for thing in self.things ]

        if thing in self.agents:
            self.agents.remove(thing)
    

class XYEnvironment(Environment):
    """This class is for environments on a 2D plane, with locations
    labelled by (x, y) points, either discrete or continuous.
    
    Agents perceive things within a radius.  Each agent in the
    environment has a .location slot which should be a location such
    as (0, 1), and a .holding slot, which should be a list of things
    that are held."""
    

    def __init__(self, width = 10, height = 10):
        super(XYEnvironment, self).__init__()
        update(self, width=width, height=height, observers=[])

    def things_near(self, location, radius = None):
        """Return all things within radius of location."""
        if radius is None:
            radius = self.perceptible_distance
        radius2 = radius * radius
        return [ thing for thing in self.things if distance2(location, thing.location) == radius2 ]

    perceptible_distance = 1

    def percept(self, agent):
        """By default, agent perceives things within a default radius."""
        return [ self.thing_percept(thing, agent) for thing in self.things_near(agent.location) ]


    def execute_action(self, agent, action):
        agent.bump = False
        if action == 'TurnRight':
            agent.heading = self.turn_heading(agent.heading, -1)
        elif action == 'TurnLeft':
            agent.heading = self.turn_heading(agent.heading, +1)
        elif action == 'Forward':
            self.move_to(agent, vector_add(agent.heading, agent.location))
        elif action == 'Release':
            if agent.holding:
                agent.holding.pop()

    def thing_percept(self, thing, agent):
        """Return the percept for this thing."""
        return thing.__class__.__name__

    def default_location(self, thing):
        return (random.choice(self.width), random.choice(self.height))

    def move_to(self, thing, destination):
        """Move a thing to a new location."""
        thing.bump = self.some_things_at(destination, Obstacle)
        if not thing.bump:
            thing.location = destination
            for o in self.observers:
                o.thing_moved(thing)

    def add_thing(self, thing, location = (1, 1)):
        super(XYEnvironment, self).add_thing(thing, location)
        thing.holding = []
        thing.held = None
        for obs in self.observers:
            obs.thing_added(thing)

    def delete_thing(self, thing):
        super(XYEnvironment, self).delete_thing(thing)
        for obs in self.observers:
            obs.thing_deleted(thing)

    def add_walls(self):
        """Put walls around the entire perimeter of the grid."""
        for x in range(self.width):
            self.add_thing(Wall(), (x, 0))
            self.add_thing(Wall(), (x, self.height - 1))

        for y in range(self.height):
            self.add_thing(Wall(), (0, y))
            self.add_thing(Wall(), (self.width - 1, y))

    def add_observer(self, observer):
        """Adds an observer to the list of observers.
        An observer is typically an EnvGUI.
        
        Each observer is notified of changes in move_to and add_thing,
        by calling the observer's methods thing_moved(thing)
        and thing_added(thing, loc)."""
        self.observers.append(observer)

    def turn_heading(self, heading, inc):
        """Return the heading to the left (inc=+1) or right (inc=-1) of heading."""
        return turn_heading(heading, inc)


class Obstacle(Thing):
    """Something that can cause a bump, preventing an agent from
    moving into the same square it's in."""
    pass


class Wall(Obstacle):
    pass


class Dirt(Thing):
    pass


class VacuumEnvironment(XYEnvironment):
    """The environment of [Ex. 2.12]. Agent perceives dirty or clean,
    and bump (into obstacle) or not; 2D discrete world of unknown size;
    performance measure is 100 for each dirt cleaned, and -1 for
    each turn taken."""

    def __init__(self, width = 10, height = 10):
        super(VacuumEnvironment, self).__init__(width, height)
        self.add_walls()

    def thing_classes(self):
        return [Wall,
         Dirt,
         ReflexVacuumAgent,
         RandomVacuumAgent,
         TableDrivenVacuumAgent,
         ModelBasedVacuumAgent]

    def percept(self, agent):
        """The percept is a tuple of ('Dirty' or 'Clean', 'Bump' or 'None').
        Unlike the TrivialVacuumEnvironment, location is NOT perceived."""
        status = if_(self.some_things_at(agent.location, Dirt), 'Dirty', 'Clean')
        bump = if_(agent.bump, 'Bump', 'None')
        return (status, bump)

    def execute_action(self, agent, action):
        if action == 'Suck':
            dirt_list = self.list_things_at(agent.location, Dirt)
            if dirt_list != []:
                dirt = dirt_list[0]
                agent.performance += 100
                self.delete_thing(dirt)
        else:
            super(VacuumEnvironment, self).execute_action(agent, action)
        if action != 'NoOp':
            agent.performance -= 1


class TrivialVacuumEnvironment(Environment):
    """This environment has two locations, A and B. Each can be Dirty
    or Clean.  The agent perceives its location and the location's
    status. This serves as an example of how to implement a simple
    Environment."""

    def __init__(self):
        super(TrivialVacuumEnvironment, self).__init__()
        self.status = {loc_A: random.choice(['Clean', 'Dirty']),
         loc_B: random.choice(['Clean', 'Dirty'])}

    def thing_classes(self):
        return [Wall,
         Dirt,
         ReflexVacuumAgent,
         RandomVacuumAgent,
         TableDrivenVacuumAgent,
         ModelBasedVacuumAgent]

    def percept(self, agent):
        """Returns the agent's location, and the location status (Dirty/Clean)."""
        return (agent.location, self.status[agent.location])

    def execute_action(self, agent, action):
        """Change agent's location and/or location's status; track performance.
        Score 10 for each dirt cleaned; -1 for each move."""
        if action == 'Right':
            agent.location = loc_B
            agent.performance -= 1
        elif action == 'Left':
            agent.location = loc_A
            agent.performance -= 1
        elif action == 'Suck':
            if self.status[agent.location] == 'Dirty':
                agent.performance += 10
            self.status[agent.location] = 'Clean'

    def default_location(self, thing):
        """Agents start in either location at random."""
        return random.choice([loc_A, loc_B])


def compare_agents(EnvFactory, AgentFactories, n = 10, steps = 1000):
    """See how well each of several agents do in n instances of an environment.
    Pass in a factory (constructor) for environments, and several for agents.
    Create n instances of the environment, and run each agent in copies of
    each one for steps. Return a list of (agent, average-score) tuples."""
    envs = [ EnvFactory() for i in range(n) ]
    return [ (A, 
    (A, steps, copy.deepcopy(envs))) for A in AgentFactories ]


def test_agent(AgentFactory, steps, envs, percepts):
    """Return the mean score of running an agent in each of the envs, for steps"""
    print ('RUN TEST AGENT')
    envs.add_thing(AgentFactory)
    #envs.run(steps)
    
    agent = AgentFactory
    agent.program(percept)
    #envs.run(steps)
    envs.runPLWumpus(agent, steps)
    #envs.runPLWumpus(steps)
    print(' ------------PLWumpus test agent KB-----------------------')
    print(agent.KB.clauses)
    #print envs.to_string()
    print('test_agent', envs)
    #print agent.KB.clauses
    return agent.performance

    def score(env):
        agent = AgentFactory()
        env.add_thing(agent)
        env.run(steps)
        print('test_agent' , env)
        return agent.performance

    #return mean(map(score, envs))
    return None

'''def test_agent(explorer, steps, wEnv):
    """Return the mean score of running an agent in each of the envs, for steps"""
    count = 0
    print 'STEPS %d' % steps
    print wEnv.to_string()
    print '\n------------------------------  --------------------------\n'
    while(count < steps - 1):
        print 'Possible actions: [quit, stop, exit] actions = [TurnRight, TurnLeft, Forward, Grab, Release, Shoot, Wait] '
        wEnv.step()
        print wEnv.percept(explorer)
        print wEnv.to_string()
        print 'Sense the environment', wEnv.percept(explorer)
        percept2 = wEnv.percept(explorer)
        pvec = explorer.raw_percepts_to_percept_vector(percept2)
        senses = [explorer.pretty_percept_vector(pvec)[5], explorer.pretty_percept_vector(pvec)[6], explorer.pretty_percept_vector(pvec)[7], explorer.pretty_percept_vector(pvec)[8], explorer.pretty_percept_vector(pvec)[9]]   
        print 'Environment', senses;
        print '\n------------------------------  --------------------------\n'
        count = count + 1 '''
        




#-------------------------------------------------------------------------------
# Wumpus World Scenarios
#-------------------------------------------------------------------------------

class WumpusWorldScenario(object):
    """
    Construct a Wumpus World Scenario
    Objects that can be added to the environment:
        Wumpus()
        Pit()
        Gold()
        Wall()
        HybridWumpusAgent(heading)  # A propositional logic Wumpus World agent
        Explorer(program, heading)  # A non-logical Wumpus World agent (mostly for debugging)
    Provides methods to load layout from file
    Provides step and run methods to run the scenario
        with the provided agent's agent_program
    """
    
    def __init__(self, layout_file=None, agent=None, objects=None,
                 width=None, height=None, entrance=None, trace=True):
        """
        layout_file := (<string: layout_file_name>, <agent>)
        """
        if agent != None and not isinstance(agent, Explorer):
            raise Exception("agent must be type Explorer, got instance of class\n" \
                            + " {0}".format(agent.__class__))
        if layout_file:
            objects, width, height, entrance = self.load_layout(layout_file)
            
        self.width, self.height = width, height
        self.entrance = entrance
        self.agent = agent
        self.objects = objects
        self.trace = trace
        self.env = self.build_world(width, height, entrance, agent, objects)

    def build_world(self, width, height, entrance, agent, objects):
        """
        Create a WumpusEnvironment with dimensions width,height
        Set the environment entrance
        objects := [(<wumpus_environment_object>, <location: (<x>,<y>) >, ...]
        """
        env = WumpusEnvironment(width, height, entrance)
        if self.trace:
            agent = wumpus_environment.TraceAgent(agent)
        agent.register_environment(env)
        env.add_thing(agent, env.entrance)
        for (obj, loc) in objects:
            env.add_thing(obj, loc)
        print env.to_string()
        print self.objects   
        return env


    def step(self):
        self.env.step()
        print
        print "Current Wumpus Environment:"
        print self.env.to_string()

    def run(self, steps = 1000):
        print self.env.to_string()
        for step in range(steps):
            if self.env.is_done():
                print "DONE."
                slist = []
                if len(self.env.agents) > 0:
                    slist += ['Final Scores:']
                for agent in self.env.agents:
                    slist.append(' {0}={1}'.format(agent, agent.performance_measure))
                    if agent.verbose:
                        if hasattr(agent, 'number_of_clauses_over_epochs'):
                            print "number_of_clauses_over_epochs:" \
                                  +" {0}".format(agent.number_of_clauses_over_epochs)
                        if hasattr(agent, 'belief_loc_query_times'):
                            print "belief_loc_query_times:" \
                                  +" {0}".format(agent.belief_loc_query_times)
                print ''.join(slist)
                return
            self.step()

    def to_string(self):
        s = "Environment width={0}, height={1}\n".format(self.width, self.height)
        s += "Initial Position: {0}\n".format(self.entrance)
        s += "Actions: {0}\n".format(self.actions)
        return s

    def pprint(self):
        print self.to_string()
        print self.env.to_string()



#-------------------------------------------------------------------------------


#------------------------------------
# examples of constructing manually-playable scenarios
# specifying objects as list or from layout file

def wscenario_5x5():
    return WumpusWorldScenario(agent = Explorer(heading='north', verbose=True),
                               objects = [(Wumpus(),(1,3)),
                                          (Pit(),(3,3)),
                                          (Pit(),(4,4)),
                                          (Pit(),(3,1)),
                                          (Gold(),(2,3))],
                               width = 5, height = 5, entrance = (2,2),
                               trace=False)


#-------------------------------------------------------------------------------
# Manual agent program
#-------------------------------------------------------------------------------

def with_manual_program(agent):
    """
    Take <agent> and replaces its agent_program with manual_program.
    manual_program waits for keyboard input and executes command.
    This uses a closure.  Three cheers for closures !!!
    (if you don't know what a closure is, read this:
       http://en.wikipedia.org/wiki/Closure_(computer_science) )
    """

    helping  = ['?', 'help']
    stopping = ['quit', 'stop', 'exit']
    actions  = ['TurnRight', 'TurnLeft', 'Forward', 'Grab', 'Release', 'Shoot', 'Wait']

    def show_commands():
        print "   The following are valid Hunt The Wumpus action:"
        print "     {0}".format(', '.join(map(lambda a: '\'{0}\''.format(a), actions)))
        print "   Enter {0} to get this command info" \
              .format(' or '.join(map(lambda a: '\'{0}\''.format(a), helping)))
        print "   Enter {0} to stop playing" \
              .format(' or '.join(map(lambda a: '\'{0}\''.format(a), stopping)))
        print "   Enter 'env' to display current wumpus environment"

    def manual_program(percept):
        print "[{0}] You perceive: {1}".format(agent.time,agent.pretty_percept_vector(percept))
        action = None
        while not action:
            val = raw_input("Enter Action ('?' for list of commands): ")
            val = val.strip()
            if val in helping:
                print
                show_commands()
                print
            elif val in stopping:
                action = 'Stop'
            elif val == 'env':
                print
                print "Current wumpus environment:"
                print agent.env.to_string()
            elif val in actions:
                action = val
            else:
                print "'{0}' is an invalid command;".format(val) \
                      + " try again (enter '?' for list of commands)"
        agent.time += 1
        return action

    agent.program = manual_program
    return agent

#-------------------------------------------------------------------------------
# Manual agent program with Knowledge Base
#-------------------------------------------------------------------------------

def with_manual_kb_program(agent):
    """
    Take <agent> and replaces its agent_program with manual_kb_program.
    Assumes the <agent> is a HybridWumpusAgent.
    (TODO: separate out logical agent from HybridWumpusAgent)
    Agent program that waits for keyboard input and executes command.
    Also provides interface for doing KB queries.
    Closures rock!
    """

    helping = ['?', 'help']
    stopping = ['quit', 'stop', 'exit']
    actions = ['TurnRight', 'TurnLeft', 'Forward', 'Grab', 'Release', 'Shoot', 'Wait']
    queries = [('qp','Query a single proposition;\n' \
                + '           E.g. \'qp B1_1\' or \'qp OK1_1_3\', \'qp HeadingWest4\''),
               ('qpl','Query a-temporal location-based proposition at all x,y locations;\n' \
                + '           E.g., \'qpl P\' runs all queries of P<x>_<y>'),
               ('qplt','Query temporal and location-based propositions at all x,y locations;\n' \
                + '           E.g., \'qplt OK 4\' runs all queries of the OK<x>_<y>_4'),
               ('q!','Run ALL queries for optionally specified time (default is current time);\n'\
                + '           (can be time consuming!)')]

    def show_commands():
        print "Available Commands:"
        print "   The following are valid Hunt The Wumpus actions:"
        print "     {0}".format(', '.join(map(lambda a: '\'{0}\''.format(a), actions)))
        print "   Enter {0} to get this command info" \
              .format(' or '.join(map(lambda a: '\'{0}\''.format(a), helping)))
        print "   Enter {0} to stop playing" \
              .format(' or '.join(map(lambda a: '\'{0}\''.format(a), stopping)))
        print "   Enter 'env' to display current wumpus environment"
        print "   Enter 'kbsat' to check if the agent's KB is satisfiable"
        print "      If the KB is NOT satisfiable, then there's a contradiction that needs fixing."
        print "      NOTE: A satisfiable KB does not mean there aren't other problems."
        print "   Enter 'save-axioms' to save all of the KB axioms to 'kb-axioms.txt'"
        print "      This will overwrite any existing 'kb-axioms.txt'"
        print "   Enter 'save-clauses' to save all of the KB clauses to text file 'kb-clauses.txt'"
        print "      This will overwrite any existing 'kb-clauses.txt'"
        print "   Enter 'props' to list all of the proposition bases"
        print "   Queries:"
        for query,desc in queries:
            print "      {0} : {1}".format(query,desc)

    def show_propositions():
        print "Proposition Bases:"
        print "   Atemporal location-based propositions (include x,y index: P<x>_<y>)"
        print "     '" + '\', \''.join(proposition_bases_atemporal_location) + '\''
        print "   Perceptual propositions (include time index: P<t>)"
        print "     '" + '\', \''.join(proposition_bases_perceptual_fluents) + '\''
        print "   Location fluent propositions (include x,y and time index: P<x>_<y>_<t>)"
        print "     '" + '\', \''.join(proposition_bases_location_fluents) + '\''
        print "   State fluent propositions (include time index: P<t>)"
        print "     '" + '\', \''.join(proposition_bases_state_fluents[:4]) + '\','
        print "     '" + '\', \''.join(proposition_bases_state_fluents[4:]) + '\''
        print "   Action propositions (include time index: P<t>)"
        print "     '" + '\', \''.join(proposition_bases_actions) + '\''

    def write_list_to_text_file(filename,list):
        outfile = file(filename, 'w')
        for item in list:
            outfile.write('{0}\n'.format(item))
        outfile.close()

    def check_kb_status():
        """
        Tests whether the agent KB is satisfiable.
        If not, that means the KB contains a contradiction that needs fixing.
        However, being satisfiable does not mean the KB is correct.
        """
        result = minisat(agent.kb.clauses)
        if result:
            print "Agent KB is satisfiable"
        else:
            print "Agent KB is NOT satisfiable!!  There is contradiction that needs fixing!"

    def simple_query(proposition):
        """
        Executes a simple query to the agent KB for specified proposition.
        """
        result = agent.kb.ask(expr(proposition))
        if result == None:
            print "{0}: Unknown!".format(proposition)
        else:
            print "{0}: {1}".format(proposition,result)

    def location_based_query(proposition_base):
        """
        Executes queries for the specified type of proposition, for
        each x,y location.
        proposition_base := as all of the propositions include in their
        name 1 or more indexes (for time and/or x,y location), the
        proposition_base is the simple string representing the base
        of the proposition witout the indexes, which are added in
        code, below.
        time := the time index of the propositions being queried
        """
        display_env = WumpusEnvironment(agent.width, agent.height)
        start_time = clock()
        print "Running queries for: {0}<x>_<y>".format(proposition_base)
        for x in range(1,agent.width+1):
            for y in range(1,agent.height+1):
                query = expr('{0}{1}_{2}'.format(proposition_base,x,y))
                result = agent.kb.ask(query)
                if result == None:
                    display_env.add_thing(Proposition(query,'?'),(x,y))
                else:
                    display_env.add_thing(Proposition(query,result),(x,y))
        end_time = clock()
        print "          >>> time elapsed while making queries:" \
              + " {0}".format(end_time-start_time)
        print display_env.to_string(agent.time,
                                    title="All {0}<x>_<y> queries".format(proposition_base))

    def location_time_based_query(proposition_base, time):
        """
        Executes queries for the specified type of proposition, for
        each x,y location, at the specified time.
        proposition_base := as all of the propositions include in their
        name 1 or more indexes (for time and/or x,y location), the
        proposition_base is the simple string representing the base
        of the proposition witout the indexes, which are added in
        code, below.
        time := the time index of the propositions being queried
        """
        display_env = WumpusEnvironment(agent.width, agent.height)
        start_time = clock()
        print "Running queries for: {0}<x>_<y>_{1}".format(proposition_base,time)
        for x in range(1,agent.width+1):
            for y in range(1,agent.height+1):
                query = expr('{0}{1}_{2}_{3}'.format(proposition_base,x,y,time))
                result = agent.kb.ask(query)
                if result == None:
                    display_env.add_thing(Proposition(query,'?'),(x,y))
                else:
                    display_env.add_thing(Proposition(query,result),(x,y))
        end_time = clock()
        print "          >>> time elapsed while making queries:" \
              + " {0}".format(end_time-start_time)
        print display_env.to_string(agent.time,
                                    title="All {0}<x>_<y>_{1} queries".format(proposition_base,
                                                                              time))

    def run_all_queries(time):
        check_kb_status()
        for p in proposition_bases_perceptual_fluents:
            simple_query(p + '{0}'.format(time))
        for p in proposition_bases_atemporal_location:
            location_based_query(p)
        for p in proposition_bases_location_fluents:
            location_time_based_query(p,time)
        for p in proposition_bases_state_fluents:
            simple_query(p + '{0}'.format(time))
        # remove the quotes below and add quotes to the following if-statement
        # in order to query all actions from time 0 to now
        '''
        print "Querying actions from time 0 to {0}".format(time)
        for p in propositions_actions:
            for t in range(time+1):
                simple_query(p + '{0}'.format(t))
        '''
        if time-1 > 0:
            print "Actions from previous time: {0}".format(time-1)
            for p in proposition_bases_actions:
                simple_query(p + '{0}'.format(time-1))
            
        print "FINISHED running all queries for time {0}".format(time)

    def manual_kb_program(percept):

        print "------------------------------------------------------------------"
        print "At time {0}".format(agent.time)
        # update current location and heading based on current KB knowledge state
        print "     HWA.infer_and_set_belief_location()"
        agent.infer_and_set_belief_location()
        print "     HWA.infer_and_set_belief_heading()"
        agent.infer_and_set_belief_heading()

        percept_sentence = agent.make_percept_sentence(percept)
        print "     HWA.agent_program(): kb.tell(percept_sentence):"
        print "         {0}".format(percept_sentence)
        agent.kb.tell(percept_sentence) # update the agent's KB based on percepts

        clauses_before = len(agent.kb.clauses)
        print "     HWA.agent_program(): Prepare to add temporal axioms"
        print "         Number of clauses in KB before: {0}".format(clauses_before)
        agent.add_temporal_axioms()
        clauses_after = len(agent.kb.clauses)
        print "         Number of clauses in KB after: {0}".format(clauses_after)
        print "         Total clauses added to KB: {0}".format(clauses_after - clauses_before)
        agent.number_of_clauses_over_epochs.append(len(agent.kb.clauses))

        action = None
        while not action:
            print "[{0}] You perceive: {1}".format(agent.time,
                                                   agent.pretty_percept_vector(percept))
            val = raw_input("Enter Action ('?' for list of commands): ")
            val = val.strip()
            if val in helping:
                print
                show_commands()
                print
            elif val in stopping:
                action = 'Stop'
            elif val in actions:
                action = val
            elif val == 'env':
                print
                print "Current wumpus environment:"
                print agent.env.to_string()
            elif val == 'props':
                print
                show_propositions()
                print
            elif val == 'kbsat':
                check_kb_status()
                print
            elif val == 'save-axioms':
                write_list_to_text_file('kb-axioms.txt',agent.kb.axioms)
                print "   Saved to 'kb-axioms.txt'"
                print
            elif val == 'save-clauses':
                write_list_to_text_file('kb-clauses.txt',agent.kb.clauses)
                print "   Saved to 'kb-clauses.txt'"
                print
            else:
                q = val.split(' ')
                if len(q) == 2 and q[0] == 'qp':
                    simple_query(q[1])
                    print
                elif len(q) == 2 and q[0] == 'qpl':
                    location_based_query(q[1])
                    print
                elif len(q) == 3 and q[0] == 'qplt':
                    location_time_based_query(q[1],q[2])
                    print
                elif q[0] == 'q!':
                    if len(q) == 2:
                        t = int(q[1])
                        run_all_queries(t)
                    else:
                        run_all_queries(agent.time)
                    print
                else:
                    print "'{0}' is an invalid command;".format(val) \
                          + " try again (enter '?' for list of commands)"
                    print

        # update KB with selected action
        agent.kb.tell(add_time_stamp(action, agent.time))

        agent.time += 1
        
        return action

    agent.program = manual_kb_program
    return agent



#-------------------------------------------------------------------------------
# Command-line interface
#-------------------------------------------------------------------------------

def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
    """
    Processes the command used to run wumpus.py from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:     python wumpus.py <options>
    EXAMPLES:  (1) python wumpus.py
                   - starts simple manual Hunt The Wumpus game
               (2) python wumpus.py -k OR python wumpus.py --kb
                   - starts simple manual Hunt The Wumpus game with
                   knowledge base and interactive queries possible
    """
    parser = OptionParser(usageStr)

    parser.add_option('-k', '--kb', action='store_true', dest='kb', default=False,
                      help=default("Instantiate a queriable knowledge base"))
    parser.add_option('-y', '--hybrid', action='store_true', dest='hybrid', default=False,
                      help=default("Run hybrid wumpus agent" \
                                   + " (takes precedence over -k option)"))
    parser.add_option('-l', '--layout', dest='layout', default=None,
                      help=default("Load layout file"))

    parser.add_option('-t', '--test', action='store_true', dest='test_minisat',
                      default=False,
                      help=default("Test connection to command-line MiniSat"))

    options, otherjunk = parser.parse_args(argv)
    
    if len(otherjunk) != 0:
        raise Exception("Command line input not understood: " + str(otherjunk))

    return options

def run_command(options):
    if options.test_minisat:
        run_minisat_test()
        return
    if options.hybrid:
        if options.layout:
            s = world_scenario_hybrid_wumpus_agent_from_layout(options.layout)
        else:
            s = wscenario_4x4_HybridWumpusAgent()
    elif options.kb:
        if options.layout:
            s = world_scenario_manual_with_kb_from_layout(options.layout)
        else:
            s = wscenario_4x4_manual_HybridWumpusAgent()
    else:
        if options.layout:
            s = world_scenario_manual_from_layout(options.layout)
        else:
            s = wscenario_4x4()
    s.run()


class Wumpus(Thing):

    def __init__(self):
        self.alive = True

    def to_string(self):
        if self.alive:
            return 'W'
        else:
            return 'X'


class Wall(Obstacle):

    def to_string(self):
        return '[]'


class Pit(Thing):

    def to_string(self):
        return 'P'


class Gold(Thing):

    def to_string(self):
        return 'G'


class Arrow(Thing):
    pass


class Explorer(Agent):

    heading_num_to_str = {0: 'north', 1: 'west', 2: 'south', 3: 'east'}
    heading_str_to_num = {'north': 0, 'west': 1, 'south': 2, 'east': 3}

    def __init__(self, program = None, heading = 'east', environment = None, verbose=True):
        """
        NOTE: AIMA Ch7 example defaults to agent initially facing east,
        which is heading=3
        """
        self.verbose = verbose
        super(Explorer, self).__init__(program)
        if isinstance(heading, str):
            heading = self.heading_str_to_num[heading]
        self.initial_heading = heading
        self.has_arrow = True
        self.has_gold = False
        self.performance_measure = 0
        if environment:
            self.register_environment(environment)

    def register_environment(self, environment):
        if self.verbose:
            "{0}.register_environment()".format(self.__class__.__name__)
        # NOTE: agent.location is the true, environment-registered location
        #       agent.location is also set by env.add_thing(agent)
        self.location = environment.entrance
        # agent.initial_location stores the original location, here same as env.entrance
        self.initial_location = environment.entrance
        # dimensions of environment
        # subtract 1 b/c env constructor always adds one for outer walls
        self.width, self.height = environment.width - 1, environment.height - 1
        self.env = environment
        self.reset()

    def reset(self):
        """
        NOTE: Eventually move belief_locaiton and belief_heading to a knowledge-based agent.
        """
        if self.verbose:
            "{0}.reset()".format(self.__class__.__name__)
        if hasattr(self,'initial_location'):
            # self.location is the true agent location in the environment
            self.location = self.initial_location
            # self.belief_locataion is location the agent believes it is in
            self.belief_location = self.initial_location
        else:
            "{0}.reset(): agent has no initial_location;".format(self.__class__.__name__)
            "     Need to first call Explorer.register_environment(env)"
        # self.heading is the true agent heading in the environment
        self.heading = self.initial_heading
        # self.belief_heading is the heading the agent believes it has
        self.belief_heading = self.initial_heading
        self.time = 0

    def heading_str(self, heading):
        """Overkill!  But once I got started, I couldn't stop making it safe...
        Ensure that heading is a valid heading 'string' (for the logic side),
        as opposed to the integer form for the WumpusEnvironment side.
        """
        if isinstance(heading,int):
            if 0 <= heading <= 3:
                return self.heading_num_to_str[heading]
            else:
                print "Not a valid heading int (0 <= heading <= 3), got: {0}".format(heading)
                sys.exit(0)
        elif isinstance(heading,str):
            headings = self.heading_str_to_num.keys()
            if heading in headings:
                return heading
            else:
                print "Not a valid heading str (one of {0}), got: {1}".format(headings,heading)
                sys.exit(0)
        else:
            print "Not a valid heading:", heading
            sys.exit(0)

    def heading_int(self, heading):
        """ Same commend in doc for heading_str applies...
        Ensure that heading is a valid integer (for the WumpusEnvironment side).
        """
        if isinstance(heading,int):
            if 0 <= heading <= 3:
                return heading
            else:
                print "Not a valid heading int (0 <= heading <= 3), got: {0}".format(heading)
                sys.exit(0)
        elif isinstance(heading,str):
            headings = self.heading_str_to_num.keys()
            if heading in headings:
                return self.heading_str_to_num[heading]
            else:
                print "Not a valid heading str (one of {0}), got: {1}".format(headings,heading)
                sys.exit(0)
        else:
            print "Not a valid heading:", heading
            sys.exit(0)

    def to_string(self):
        """
        String representation of TRUE agent heading
        NOTE: This should really be the responsibility of the environment,
              refactor at some point
        """
        if self.heading == 0:
            return '^'
        if self.heading == 1:
            return '<'
        if self.heading == 2:
            return 'v'
        if self.heading == 3:
            return '>'

    def pretty_percept_vector(self, pvec):
        """ percept_vector: [<Stench?>, <Breeze?>, <Glitter?>, <Bump?>, <Scream?>] """
        percept_vector = [ 'None' for i in range(len(pvec)) ]
        if pvec[0]: percept_vector.append('Stench')
        else: percept_vector.append('~Stench')
        if pvec[1]: percept_vector.append('Breeze')
        else: percept_vector.append('~Breeze')
        if pvec[2]: percept_vector.append('Glitter')
        else: percept_vector.append('~Glitter')
        if pvec[3]: percept_vector.append('Bump')
        else: percept_vector.append('~Bump')
        if pvec[4]: percept_vector.append('Scream')
        else: percept_vector.append('~Scream')
        return percept_vector

    def raw_percepts_to_percept_vector(self, percepts):
        """
        raw percepts are: [<time_step>,
                           <Things in range>...,
                           <exogenous events ('Bump', 'Scream')>...]
        percept_vector: [<Stench?>, <Breeze?>, <Glitter?>, <Bump?>, <Scream?>]
        """
        percept_vector = [ None for i in range(5) ]
        # print 'raw percepts:', percepts
        for rawp in percepts:
            if rawp == 'Wumpus':
                percept_vector[0] = True
            if rawp == 'Pit':
                percept_vector[1] = True
            if rawp == 'Gold':
                percept_vector[2] = True
            if rawp == 'Bump':
                percept_vector[3] = True
            if rawp == 'Scream':
                percept_vector[4] = True

        return percept_vector

def TraceAgent(agent):
    """
    Wrap the agent's program to print its input and output. This will let
    you see what the agent is doing in the environment.
    
    This is still used in wumpus.WumpusWorldEnvironment.build_world,
    although it is now largley redundant b/c WumpusEnvironment has a
    verbose flag, and the with_manual*_program wrapper do lots of
    printing of state.
    """
    old_program = agent.program

    def new_program(percept):
        action = old_program(percept)
        print '%s perceives %s and does %s' % (agent,
                                               agent.pretty_percept_vector(percept),
                                               action)
        return action

    agent.program = new_program
    return agent

#-----Wumpus Environment-------------------------------
class WumpusEnvironment(XYEnvironment):

    def __init__(self, width = 5, height = 5, entrance = (1, 1)):
        """ NOTE: range from 1 to {width or height} contains map,
        anything outside, 0 and {width+1 or height+1} becomes a wall """
        super(WumpusEnvironment, self).__init__(width + 1, height + 1)
        self.entrance = entrance
        self.add_walls()
        #self.add_things()
        self.time_step = 0
        self.done = False
        self.global_percept_events = []
        #self.add_thing(Wumpus(),(1,3))
        #self.add_thing(Pit(),(3,3))
        #self.add_thing(Pit(),(3,1))
        #self.add_thing(Gold(),(2,3))
        #self.add_thing(Pit(),(4,4))
        
   

    def thing_classes(self):
        return [agents.Wall,
                Pit,
                Arrow,
                Gold,
                Wumpus,
                Explorer]
                

    def exogenous_change(self):
        """ Handle special outcomes """
        for agent in self.agents:
            colocated_wumpi = [ wumpus.is_alive()
                                for wumpus in self.list_things_at(agent.location,
                                                                  tclass=Wumpus) ]
            colocated_pit = self.list_things_at(agent.location, tclass=Pit)
            if any(colocated_wumpi):
                print 'A Wumpus ate {0}!'.format(agent)
                agent.performance_measure -= 1000
                self.done = True
            elif colocated_pit:
                print '{0} fell into a bottomless pit!'.format(agent)
                agent.performance_measure -= 1000
                self.done = True

    def is_done(self):
        return self.done or not any((agent.is_alive() for agent in self.agents))

    def step(self):
        super(WumpusEnvironment, self).step()
        self.time_step += 1

    def turn_heading(self, heading, inc):
        """ Return the heading to the left (inc=+1) or right (inc=-1) of heading.
        Only 4 directions, so mod(heading+inc,4) """
        return (heading + inc) % 4

    def heading_to_vector(self, heading):
        """ Convert heading into vector that can be added to location
        if agent moves Forward """
        if heading == 0:
            v = (0, 1)
        elif heading == 1:
            v = (-1, 0)
        elif heading == 2:
            v = (0, -1)
        elif heading == 3:
            v = (1, 0)
        return v
        
    def heading_to_str(self, heading):
        """ Convert heading into vector that can be added to location
        if agent moves Forward """
        if heading == 0:
            v = 'North'
        elif heading == 1:
            v = 'West'
        elif heading == 2:
            v = 'South'
        elif heading == 3:
            v = 'East'
        return v
        
    def percept(self, agent):
        """By default, agent perceives things within a default radius."""
        return [ self.thing_percept(thing, agent) for thing in self.things_near(agent.location) ]

    def percept2(self, agent):
        """ Each percept is a list beginning with the time_step (integer) """
        percepts = [self.time_step]
        for thing in self.things_near(agent.location):
            if isinstance(thing, Gold):
                if agent.location == thing.location:
                    percepts.append('Gold')
            elif isinstance(thing, Wumpus):
                if agent.location == thing.location:
                    percepts.append('Wumpus')
            else:
                percepts.append(self.thing_percept(thing, agent))

        if agent.bump:
            percepts.append('Bump')
        percepts += self.global_percept_events
        agent.bump = False
        self.global_percept_events = []
        return agent.raw_percepts_to_percept_vector(percepts)

    def execute_action(self, agent, action):
        """ Execute action taken by agent """
        agent.bump = False
        agent.performance_measure -= 1
        
        if action == 'TurnRight':
            agent.heading = self.turn_heading(agent.heading, -1)
        elif action == 'TurnLeft':
            agent.heading = self.turn_heading(agent.heading, +1)
        elif action == 'Forward':
            self.move_to(agent, vector_add(self.heading_to_vector(agent.heading),
                                           agent.location))
        elif action == 'Grab':
            if self.some_things_at(agent.location, tclass=Gold):
                try:
                    gold = self.list_things_at(agent.location, tclass=Gold)[0]
                    agent.has_gold = True
                    self.delete_thing(gold)
                except:
                    print "Error: Gold should be here, but couldn't find it!"
                    print 'All things:', self.list_things_at(agent.location)
                    print 'Gold?:', self.list_things_at(agent.location, tclass=Gold)
                    sys.exit(-1)

        elif action == 'Release':
            if agent.location == self.entrance:
                if agent.has_gold:
                    agent.performance_measure += 1000
                self.done = True
        elif action == 'Shoot':
            if agent.has_arrow:
                agent.has_arrow = False
                agent.performance_measure -= 10
                self.shoot_arrow(agent)
        elif action == 'Stop':
            self.done = True
        
        print '\nCurrent Location: ', agent.location
        print 'Heading: ', self.heading_to_str(agent.heading)
        print 'Reminder- Start Location:', self.entrance
        print ''
        print 'Percepts:'
        

    def shoot_arrow(self, agent):
        dvec = self.heading_to_vector(agent.heading)
        aloc = agent.location
        while True:
            aloc = vector_add(dvec, aloc)
            if self.some_things_at(aloc, tclass=Wumpus):
                try:
                    poor_wumpus = self.list_things_at(aloc, tclass=Wumpus)[0]
                    poor_wumpus.alive = False
                    self.global_percept_events.append('Scream')
                except:
                    print "Error: Wumpus should be here, but couldn't find it!"
                    print 'All things:', aloc, self.list_things_at(aloc)
                    print 'Wumpus?:', aloc, self.list_things_at(aloc, tclass=Wumpus)
                    sys.exit(-1)

                break
            if self.some_things_at(aloc, tclass=Wall):
                break
            if 0 > aloc[0] > self.width or 0 > aloc[1] > self.height:
                break
                
    def runPLWumpus(self, agent, steps = 1000):
        """Run the Environment for given number of time steps."""
        for step in range(steps):
            if self.is_done():
                return
            self.step()
            agent.program(percept)


    def run_verbose(self, steps = 10):
        """Run environment while displaying ascii map, for given number of steps."""
        for step in range(steps):
            if self.is_done():
                print 'Done, stopping.'
                print self.to_string()
                return
            print self.to_string()
            self.step()
            

    def add_walls(self):
        """Put walls around the entire perimeter of the grid."""
        for x in range(self.width + 1):
            if not self.some_things_at((x, 0), Wall):
                self.add_thing(Wall(), (x, 0))
            if not self.some_things_at((x, self.height), Wall):
                self.add_thing(Wall(), (x, self.height))

        for y in range(self.height + 1):
            if not self.some_things_at((0, y), Wall):
                self.add_thing(Wall(), (0, y))
            if not self.some_things_at((self.width, y), Wall):
                self.add_thing(Wall(), (self.width, y))
        #self.add_thing(Wumpus(),(1,3))
        #self.add_thing(Pit(),(3,3))
        #self.add_thing(Pit(),(3,1))
        #self.add_thing(Gold(),(2,3))
        #self.add_thing(Pit(),(4,4))

    def max_cell_print_len(self):
        """Find the max print-size of all cells"""
        m = 0
        for r in range(1, self.height + 1):
            for c in range(1, self.width + 1):
                l = 0
                for item in self.list_things_at((r, c)):
                    #print 'max_cell_print_len:', item
                    l += len(item.to_string())
                if l > m:
                    m = l
        return m

    def to_string(self, t = None, title = None):
        """ Awkward implementation of quick-n-dirty ascii display of Wumpus Environment
        Uses R&N AIMA roome coordinates: (0,0) is bottom-left in ascii display """
        if title:
            print title
        column_width = self.max_cell_print_len()
        cell_hline = [ '-' for i in range(column_width + 2) ] + ['|']
        cell_hline = ''.join(cell_hline)
        hline = ['|'] + [ cell_hline for i in range(self.width + 1) ] + ['\n']
        hline = ''.join(hline)
        slist = []
        if len(self.agents) > 0:
            slist += ['Scores:']
        for agent in self.agents:
            slist.append(' {0}={1}'.format(agent, agent.performance_measure))

        if len(self.agents) > 0:
            slist.append('\n')
        for c in range(0, self.width + 1):
            spacer = ''.join([ ' ' for i in range(column_width - 1) ])
            slist.append('  {0}{1} '.format(c, spacer))

        slist.append('   time_step={0}'.format(t if t else self.time_step))
        slist.append('\n')
        slist.append(hline)
        for r in range(self.height, -1, -1):
            for c in range(0, self.width + 1):
                things_at = self.list_things_at((c, r))
                cell_width = 0
                for thing_at in things_at:
                    cell_width += len(thing_at.to_string())

                spacer = ''.join([ ' ' for i in range(column_width - cell_width) ])
                slist.append('| ')
                for thing in things_at:
                    slist.append(thing.to_string())

                slist.append(spacer + ' ')

            slist.append('| {0}\n'.format(r))
            slist.append(hline)

        return ''.join(slist)

    def test_agent(AgentFactory, steps, envs):
        "Return the mean score of running an agent in each of the envs, for steps"
        total = 0
        for env in envs:
            agent = AgentFactory()
            env.add_object(agent)
            env.run(steps)
            total += agent.performance
        return float(total)/len(envs)
#______________________________________________________________________________

class KB(object):
    """A knowledge base to which you can tell and ask sentences.
    To create a KB, first subclass this class and implement
    tell, ask_generator, and retract.  Why ask_generator instead of ask?
    The book is a bit vague on what ask means --
    For a Propositional Logic KB, ask(P & Q) returns True or False, but for an
    FOL KB, something like ask(Brother(x, y)) might return many substitutions
    such as {x: Cain, y: Abel}, {x: Abel, y: Cain}, {x: George, y: Jeb}, etc.
    So ask_generator generates these one at a time, and ask either returns the
    first one or returns False."""

    def __init__(self, sentence=None):
        abstract

    def tell(self, sentence):
        "Add the sentence to the KB."
        abstract

    def ask(self, query):
        """Return a substitution that makes the query true, or,
        failing that, return False."""
        for result in self.ask_generator(query):
            return result
        return False

    def ask_generator(self, query):
        "Yield all the substitutions that make query true."
        abstract

    def retract(self, sentence):
        "Remove sentence from the KB."
        abstract


class PropKB(KB):
    "A KB for propositional logic. Inefficient, with no indexing."

    def __init__(self, sentence=None):
        self.clauses = []
        if sentence:
            self.tell(sentence)

    def tell(self, sentence):
        "Add the sentence's clauses to the KB."
        self.clauses.extend(conjuncts(to_cnf(sentence)))

    def ask_generator(self, query):
        "Yield the empty substitution if KB implies query; else nothing."
        if tt_entails(Expr('&', *self.clauses), query):
            yield {}

    def retract(self, sentence):
        "Remove the sentence's clauses from the KB."
        for c in conjuncts(to_cnf(sentence)):
            if c in self.clauses:
                self.clauses.remove(c)

#______________________________________________________________________________

def KB_AgentProgram(KB):
    """A generic logical knowledge-based agent program. [Fig. 7.1]"""
    steps = itertools.count()

    def program(percept):
        t = steps.next()
        KB.tell(make_percept_sentence(percept, t))
        action = KB.ask(make_action_query(t))
        KB.tell(make_action_sentence(action, t))
        return action

    def make_percept_sentence(self, percept, t):
        return Expr("Percept")(percept, t)

    def make_action_query(self, t):
        return expr("ShouldDo(action, %d)" % t)

    def make_action_sentence(self, action, t):
        return Expr("Did")(action[expr('action')], t)

    return program

#______________________________________________________________________________

class Expr:

    def __init__(self, op, *args):
        "Op is a string or number; args are Exprs (or are coerced to Exprs)."
        assert isinstance(op, str) or (isnumber(op) and not args)
        self.op = num_or_str(op)
        self.args = map(expr, args) ## Coerce args to Exprs

    def __call__(self, *args):
        """Self must be a symbol with no args, such as Expr('F').  Create a new
        Expr with 'F' as op and the args as arguments."""
        assert is_symbol(self.op) and not self.args
        return Expr(self.op, *args)

    def __repr__(self):
        "Show something like 'P' or 'P(x, y)', or '~P' or '(P | Q | R)'"
        if not self.args:         # Constant or proposition with arity 0
            return str(self.op)
        elif is_symbol(self.op):  # Functional or propositional operator
            return '%s(%s)' % (self.op, ', '.join(map(repr, self.args)))
        elif len(self.args) == 1: # Prefix operator
            return self.op + repr(self.args[0])
        else:                     # Infix operator
            return '(%s)' % (' '+self.op+' ').join(map(repr, self.args))

    def __eq__(self, other):
        """x and y are equal iff their ops and args are equal."""
        return (other is self) or (isinstance(other, Expr)
            and self.op == other.op and self.args == other.args)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        "Need a hash method so Exprs can live in dicts."
        return hash(self.op) ^ hash(tuple(self.args))

    # See http://www.python.org/doc/current/lib/module-operator.html
    # Not implemented: not, abs, pos, concat, contains, *item, *slice
    def __lt__(self, other):     return Expr('<',  self, other)
    def __le__(self, other):     return Expr('<=', self, other)
    def __ge__(self, other):     return Expr('>=', self, other)
    def __gt__(self, other):     return Expr('>',  self, other)
    def __add__(self, other):    return Expr('+',  self, other)
    def __sub__(self, other):    return Expr('-',  self, other)
    def __and__(self, other):    return Expr('&',  self, other)
    def __div__(self, other):    return Expr('/',  self, other)
    def __truediv__(self, other):return Expr('/',  self, other)
    def __invert__(self):        return Expr('~',  self)
    def __lshift__(self, other): return Expr('<<', self, other)
    def __rshift__(self, other): return Expr('>>', self, other)
    def __mul__(self, other):    return Expr('*',  self, other)
    def __neg__(self):           return Expr('-',  self)
    def __or__(self, other):     return Expr('|',  self, other)
    def __pow__(self, other):    return Expr('**', self, other)
    def __xor__(self, other):    return Expr('^',  self, other)
    def __mod__(self, other):    return Expr('<=>',  self, other)



def expr(s):
    """Create an Expr representing a logic expression by parsing the input
    string. Symbols and numbers are automatically converted to Exprs.
    In addition you can use alternative spellings of these operators:
      'x ==> y'   parses as   (x >> y)    # Implication
      'x <== y'   parses as   (x << y)    # Reverse implication
      'x <=> y'   parses as   (x % y)     # Logical equivalence
      'x =/= y'   parses as   (x ^ y)     # Logical disequality (xor)
    But BE CAREFUL; precedence of implication is wrong. expr('P & Q ==> R & S')
    is ((P & (Q >> R)) & S); so you must use expr('(P & Q) ==> (R & S)').
    >>> expr('P <=> Q(1)')
    (P <=> Q(1))
    >>> expr('P & Q | ~R(x, F(x))')
    ((P & Q) | ~R(x, F(x)))
    """
    if isinstance(s, Expr): return s
    if isnumber(s): return Expr(s)
    ## Replace the alternative spellings of operators with canonical spellings
    s = s.replace('==>', '>>').replace('<==', '<<')
    s = s.replace('<=>', '%').replace('=/=', '^')
    ## Replace a symbol or number, such as 'P' with 'Expr("P")'
    s = re.sub(r'([a-zA-Z0-9_.]+)', r'Expr("\1")', s)
    ## Now eval the string.  (A security hole; do not use with an adversary.)
    return eval(s, {'Expr':Expr})

def is_symbol(s):
    "A string s is a symbol if it starts with an alphabetic char."
    return isinstance(s, str) and s[:1].isalpha()

def is_var_symbol(s):
    "A logic variable symbol is an initial-lowercase string."
    return is_symbol(s) and s[0].islower()

def is_prop_symbol(s):
    """A proposition logic symbol is an initial-uppercase string other than
    TRUE or FALSE."""
    return is_symbol(s) and s[0].isupper() and s != 'TRUE' and s != 'FALSE'

def variables(s):
    """Return a set of the variables in expression s.
    >>> ppset(variables(F(x, A, y)))
    set([x, y])
    >>> ppset(variables(F(G(x), z)))
    set([x, z])
    >>> ppset(variables(expr('F(x, x) & G(x, y) & H(y, z) & R(A, z, z)')))
    set([x, y, z])
    """
    result = set([])
    def walk(s):
        if is_variable(s):
            result.add(s)
        else:
            for arg in s.args:
                walk(arg)
    walk(s)
    return result

def is_definite_clause(s):
    """returns True for exprs s of the form A & B & ... & C ==> D,
    where all literals are positive.  In clause form, this is
    ~A | ~B | ... | ~C | D, where exactly one clause is positive.
    >>> is_definite_clause(expr('Farmer(Mac)'))
    True
    >>> is_definite_clause(expr('~Farmer(Mac)'))
    False
    >>> is_definite_clause(expr('(Farmer(f) & Rabbit(r)) ==> Hates(f, r)'))
    True
    >>> is_definite_clause(expr('(Farmer(f) & ~Rabbit(r)) ==> Hates(f, r)'))
    False
    >>> is_definite_clause(expr('(Farmer(f) | Rabbit(r)) ==> Hates(f, r)'))
    False
    """
    if is_symbol(s.op):
        return True
    elif s.op == '>>':
        antecedent, consequent = s.args
        return (is_symbol(consequent.op)
                and every(lambda arg: is_symbol(arg.op), conjuncts(antecedent)))
    else:
        return False

def parse_definite_clause(s):
    "Return the antecedents and the consequent of a definite clause."
    assert is_definite_clause(s)
    if is_symbol(s.op):
        return [], s
    else:
        antecedent, consequent = s.args
        return conjuncts(antecedent), consequent

## Useful constant Exprs used in examples and code:
TRUE, FALSE, ZERO, ONE, TWO = map(Expr, ['TRUE', 'FALSE', 0, 1, 2])
A, B, C, F, G, P, Q, x, y, z  = map(Expr, 'ABCFGPQxyz')

#______________________________________________________________________________

def tt_entails(kb, alpha):
    """Does kb entail the sentence alpha? Use truth tables. For propositional
    kb's and sentences. [Fig. 7.10]
    >>> tt_entails(expr('P & Q'), expr('Q'))
    True
    """
    assert not variables(alpha)
    return tt_check_all(kb, alpha, prop_symbols(kb & alpha), {})

def tt_check_all(kb, alpha, symbols, model):
    "Auxiliary routine to implement tt_entails."
    if not symbols:
        if pl_true(kb, model):
            result = pl_true(alpha, model)
            assert result in (True, False)
            return result
        else:
            return True
    else:
        P, rest = symbols[0], symbols[1:]
        return (tt_check_all(kb, alpha, rest, extend(model, P, True)) and
                tt_check_all(kb, alpha, rest, extend(model, P, False)))

def prop_symbols(x):
    "Return a list of all propositional symbols in x."
    if not isinstance(x, Expr):
        return []
    elif is_prop_symbol(x.op):
        return [x]
    else:
        return list(set(symbol for arg in x.args
                        for symbol in prop_symbols(arg)))

def tt_true(alpha):
    """Is the propositional sentence alpha a tautology? (alpha will be
    coerced to an expr.)
    >>> tt_true(expr("(P >> Q) <=> (~P | Q)"))
    True
    """
    return tt_entails(TRUE, expr(alpha))

def pl_true(exp, model={}):
    """Return True if the propositional logic expression is true in the model,
    and False if it is false. If the model does not specify the value for
    every proposition, this may return None to indicate 'not obvious';
    this may happen even when the expression is tautological."""
    op, args = exp.op, exp.args
    if exp == TRUE:
        return True
    elif exp == FALSE:
        return False
    elif is_prop_symbol(op):
        return model.get(exp)
    elif op == '~':
        p = pl_true(args[0], model)
        if p is None: return None
        else: return not p
    elif op == '|':
        result = False
        for arg in args:
            p = pl_true(arg, model)
            if p is True: return True
            if p is None: result = None
        return result
    elif op == '&':
        result = True
        for arg in args:
            p = pl_true(arg, model)
            if p is False: return False
            if p is None: result = None
        return result
    p, q = args
    if op == '>>':
        return pl_true(~p | q, model)
    elif op == '<<':
        return pl_true(p | ~q, model)
    pt = pl_true(p, model)
    if pt is None: return None
    qt = pl_true(q, model)
    if qt is None: return None
    if op == '<=>':
        return pt == qt
    elif op == '^':
        return pt != qt
    else:
        raise ValueError, "illegal operator in logic expression" + str(exp)

#______________________________________________________________________________

## Convert to Conjunctive Normal Form (CNF)

def to_cnf(s):
    """Convert a propositional logical sentence s to conjunctive normal form.
    That is, to the form ((A | ~B | ...) & (B | C | ...) & ...) [p. 253]
    >>> to_cnf("~(B|C)")
    (~B & ~C)
    >>> to_cnf("B <=> (P1|P2)")
    ((~P1 | B) & (~P2 | B) & (P1 | P2 | ~B))
    >>> to_cnf("a | (b & c) | d")
    ((b | a | d) & (c | a | d))
    >>> to_cnf("A & (B | (D & E))")
    (A & (D | B) & (E | B))
    >>> to_cnf("A | (B | (C | (D & E)))")
    ((D | A | B | C) & (E | A | B | C))
    """
    if isinstance(s, str): s = expr(s)
    s = eliminate_implications(s) # Steps 1, 2 from p. 253
    s = move_not_inwards(s) # Step 3
    return distribute_and_over_or(s) # Step 4

def eliminate_implications(s):
    """Change >>, <<, and <=> into &, |, and ~. That is, return an Expr
    that is equivalent to s, but has only &, |, and ~ as logical operators.
    >>> eliminate_implications(A >> (~B << C))
    ((~B | ~C) | ~A)
    >>> eliminate_implications(A ^ B)
    ((A & ~B) | (~A & B))
    """
    if not s.args or is_symbol(s.op): return s     ## (Atoms are unchanged.)
    args = map(eliminate_implications, s.args)
    a, b = args[0], args[-1]
    if s.op == '>>':
        return (b | ~a)
    elif s.op == '<<':
        return (a | ~b)
    elif s.op == '<=>':
        return (a | ~b) & (b | ~a)
    elif s.op == '^':
        assert len(args) == 2   ## TODO: relax this restriction
        return (a & ~b) | (~a & b)
    else:
        assert s.op in ('&', '|', '~')
        return Expr(s.op, *args)

def move_not_inwards(s):
    """Rewrite sentence s by moving negation sign inward.
    >>> move_not_inwards(~(A | B))
    (~A & ~B)
    >>> move_not_inwards(~(A & B))
    (~A | ~B)
    >>> move_not_inwards(~(~(A | ~B) | ~~C))
    ((A | ~B) & ~C)
    """
    if s.op == '~':
        NOT = lambda b: move_not_inwards(~b)
        a = s.args[0]
        if a.op == '~': return move_not_inwards(a.args[0]) # ~~A ==> A
        if a.op =='&': return associate('|', map(NOT, a.args))
        if a.op =='|': return associate('&', map(NOT, a.args))
        return s
    elif is_symbol(s.op) or not s.args:
        return s
    else:
        return Expr(s.op, *map(move_not_inwards, s.args))

def distribute_and_over_or(s):
    """Given a sentence s consisting of conjunctions and disjunctions
    of literals, return an equivalent sentence in CNF.
    >>> distribute_and_over_or((A & B) | C)
    ((A | C) & (B | C))
    """
    if s.op == '|':
        s = associate('|', s.args)
        if s.op != '|':
            return distribute_and_over_or(s)
        if len(s.args) == 0:
            return FALSE
        if len(s.args) == 1:
            return distribute_and_over_or(s.args[0])
        conj = find_if((lambda d: d.op == '&'), s.args)
        if not conj:
            return s
        others = [a for a in s.args if a is not conj]
        rest = associate('|', others)
        return associate('&', [distribute_and_over_or(c|rest)
                               for c in conj.args])
    elif s.op == '&':
        return associate('&', map(distribute_and_over_or, s.args))
    else:
        return s

def associate(op, args):
    """Given an associative op, return an expression with the same
    meaning as Expr(op, *args), but flattened -- that is, with nested
    instances of the same op promoted to the top level.
    >>> associate('&', [(A&B),(B|C),(B&C)])
    (A & B & (B | C) & B & C)
    >>> associate('|', [A|(B|(C|(A&B)))])
    (A | B | C | (A & B))
    """
    args = dissociate(op, args)
    if len(args) == 0:
        return _op_identity[op]
    elif len(args) == 1:
        return args[0]
    else:
        return Expr(op, *args)

_op_identity = {'&':TRUE, '|':FALSE, '+':ZERO, '*':ONE}

def dissociate(op, args):
    """Given an associative op, return a flattened list result such
    that Expr(op, *result) means the same as Expr(op, *args)."""
    result = []
    def collect(subargs):
        for arg in subargs:
            if arg.op == op: collect(arg.args)
            else: result.append(arg)
    collect(args)
    return result

def conjuncts(s):
    """Return a list of the conjuncts in the sentence s.
    >>> conjuncts(A & B)
    [A, B]
    >>> conjuncts(A | B)
    [(A | B)]
    """
    return dissociate('&', [s])

def disjuncts(s):
    """Return a list of the disjuncts in the sentence s.
    >>> disjuncts(A | B)
    [A, B]
    >>> disjuncts(A & B)
    [(A & B)]
    """
    return dissociate('|', [s])

#______________________________________________________________________________

def pl_resolution(KB, alpha):
    "Propositional-logic resolution: say if alpha follows from KB. [Fig. 7.12]"
    clauses = KB.clauses + conjuncts(to_cnf(~alpha))
    new = set()
    while True:
        n = len(clauses)

        print 'num_clauses:', n

        pairs = [(clauses[i], clauses[j])
                 for i in range(n) for j in range(i+1, n)]

        print 'pairs:', len(pairs)

        pairs_count = 0
        for (ci, cj) in pairs:
            pairs_count += 1
            if pairs_count % 10000 == 0: print '   ', pairs_count
            resolvents = pl_resolve(ci, cj)
            if FALSE in resolvents: return True
            new = new.union(set(resolvents))
        if new.issubset(set(clauses)): return False

        print 'new:', len(new)
        
        for c in new:
            if c not in clauses: clauses.append(c)

def pl_resolve(ci, cj):
    """Return all clauses that can be obtained by resolving clauses ci and cj.
    >>> for res in pl_resolve(to_cnf(A|B|C), to_cnf(~B|~C|F)):
    ...    ppset(disjuncts(res))
    set([A, C, F, ~C])
    set([A, B, F, ~B])
    """
    clauses = []
    for di in disjuncts(ci):
        for dj in disjuncts(cj):
            if di == ~dj or ~di == dj:
                dnew = unique(removeall(di, disjuncts(ci)) +
                              removeall(dj, disjuncts(cj)))
                clauses.append(associate('|', dnew))
    return clauses

#______________________________________________________________________________

class PropDefiniteKB(PropKB):
    "A KB of propositional definite clauses."

    def tell(self, sentence):
        "Add a definite clause to this KB."
        assert is_definite_clause(sentence), "Must be definite clause"
        self.clauses.append(sentence)

    def ask_generator(self, query):
        "Yield the empty substitution if KB implies query; else nothing."
        if pl_fc_entails(self.clauses, query):
            yield {}

    def retract(self, sentence):
        self.clauses.remove(sentence)

    def clauses_with_premise(self, p):
        """Return a list of the clauses in KB that have p in their premise.
        This could be cached away for O(1) speed, but we'll recompute it."""
        return [c for c in self.clauses
                if c.op == '>>' and p in conjuncts(c.args[0])]

def pl_fc_entails(KB, q):
    """Use forward chaining to see if a PropDefiniteKB entails symbol q.
    [Fig. 7.15]
    >>> pl_fc_entails(Fig[7,15], expr('Q'))
    True
    """
    count = dict([(c, len(conjuncts(c.args[0]))) for c in KB.clauses
                                                 if c.op == '>>'])
    inferred = DefaultDict(False)
    agenda = [s for s in KB.clauses if is_prop_symbol(s.op)]
    while agenda:
        p = agenda.pop()
        if p == q: return True
        if not inferred[p]:
            inferred[p] = True
            for c in KB.clauses_with_premise(p):
                count[c] -= 1
                if count[c] == 0:
                    agenda.append(c.args[1])
    return False

## Wumpus World example [Fig. 7.13]
Fig[7,13] = expr("(B11 <=> (P12 | P21))  &  ~B11")

## Propositional Logic Forward Chaining example [Fig. 7.16]
Fig[7,15] = PropDefiniteKB()
for s in "P>>Q   (L&M)>>P   (B&L)>>M   (A&P)>>L   (A&B)>>L   A   B".split():
    Fig[7,15].tell(expr(s))

#______________________________________________________________________________
# DPLL-Satisfiable [Fig. 7.17]

def dpll_satisfiable(s):
    """Check satisfiability of a propositional sentence.
    This differs from the book code in two ways: (1) it returns a model
    rather than True when it succeeds; this is more useful. (2) The
    function find_pure_symbol is passed a list of unknown clauses, rather
    than a list of all clauses and the model; this is more efficient.
    >>> ppsubst(dpll_satisfiable(A&~B))
    {A: True, B: False}
    >>> dpll_satisfiable(P&~P)
    False
    """
    clauses = conjuncts(to_cnf(s))
    symbols = prop_symbols(s)

    print '  >>> Got clauses (',len(clauses),') and symbols (', len(symbols), ')'
    print '  >>> starting dpll proper'
    
    return dpll(clauses, symbols, {})

def dpll(clauses, symbols, model):
    "See if the clauses are true in a partial model."
    unknown_clauses = [] ## clauses with an unknown truth value
    for c in clauses:
        val =  pl_true(c, model)
        if val == False:
            return False
        if val != True:
            unknown_clauses.append(c)
    if not unknown_clauses:
        return model
    P, value = find_pure_symbol(symbols, unknown_clauses)
    if P:
        return dpll(clauses, removeall(P, symbols), extend(model, P, value))
    P, value = find_unit_clause(clauses, model)
    if P:
        return dpll(clauses, removeall(P, symbols), extend(model, P, value))
    P, symbols = symbols[0], symbols[1:]
    return (dpll(clauses, symbols, extend(model, P, True)) or
            dpll(clauses, symbols, extend(model, P, False)))

def find_pure_symbol(symbols, clauses):
    """Find a symbol and its value if it appears only as a positive literal
    (or only as a negative) in clauses.
    >>> find_pure_symbol([A, B, C], [A|~B,~B|~C,C|A])
    (A, True)
    """
    for s in symbols:
        found_pos, found_neg = False, False
        for c in clauses:
            if not found_pos and s in disjuncts(c): found_pos = True
            if not found_neg and ~s in disjuncts(c): found_neg = True
        if found_pos != found_neg: return s, found_pos
    return None, None

def find_unit_clause(clauses, model):
    """A unit clause has only 1 variable that is not bound in the model.
    >>> find_unit_clause([A|B|C, B|~C, A|~B], {A:False})    # CTM: A:True -> A:False
    (B, False)
    """
    for clause in clauses:
        num_not_in_model = 0
        true_literal_in_clause = False # CTM
        for literal in disjuncts(clause):
            sym = literal_symbol(literal)
            
            # CTM: Ensure all already assigned variables lead to literals that are false!
            # (If the literals are true, then the remaining vars could lead to
            # true or false literals)
            if sym in model:
                val = model[sym]
                if (not val and literal.op == '~') or (val and literal.op != '~'):
                    true_literal_in_clause = True
                    
            else:
                num_not_in_model += 1
                P, value = sym, (literal.op != '~')
        if num_not_in_model == 1 and not true_literal_in_clause: # CTM
            return P, value
    return None, None


def literal_symbol(literal):
    """The symbol in this literal (without the negation).
    >>> literal_symbol(P)
    P
    >>> literal_symbol(~P)
    P
    """
    if literal.op == '~':
        return literal.args[0]
    else:
        return literal

#______________________________________________________________________________
# Walk-SAT [Fig. 7.18]

def WalkSAT(clauses, p=0.5, max_flips=10000):
    ## model is a random assignment of true/false to the symbols in clauses
    ## See ~/aima1e/print1/manual/knowledge+logic-answers.tex ???
    model = dict([(s, random.choice([True, False]))
                 for s in prop_symbols(clauses)])
    for i in range(max_flips):
        satisfied, unsatisfied = [], []
        for clause in clauses:
            if_(pl_true(clause, model), satisfied, unsatisfied).append(clause)
        if not unsatisfied: ## if model satisfies all the clauses
            return model
        clause = random.choice(unsatisfied)
        if probability(p):
            sym = random.choice(prop_symbols(clause))
        else:
            ## Flip the symbol in clause that maximizes number of sat. clauses
            raise NotImplementedError
        model[sym] = not model[sym]

#______________________________________________________________________________

'''
class HybridWumpusAgent(agents.Agent):
    "An agent for the wumpus world that does logical inference. [Fig. 7.19]"""
    def __init__(self):
        unimplemented()

def plan_route(current, goals, allowed):
    unimplemented()
'''

#______________________________________________________________________________

def SAT_plan(init, transition, goal, t_max, SAT_solver=None # CTM: dpll_satisfiable
             ):
    "[Fig. 7.22]"
    for t in range(t_max):
        cnf = translate_to_SAT(init, transition, goal, t)
        model = SAT_solver(cnf)
        if model is not False:
            return extract_solution(model)
    return None

def translate_to_SAT(init, transition, goal, t):
    unimplemented()

def extract_solution(model):
    unimplemented()

#______________________________________________________________________________


# PL-Wumpus-Agent [Fig. 7.19]
class PLWumpusAgent(Agent):
    "An agent for the wumpus world that does logical inference. [Fig. 7.19]"""
    def __init__(self):
        self.KB = FOLKB()
        self.x, self.y, self.orientation = 2, 3, (1, 0)
        self.visited = set() ## squares already visited
        self.action = None
        self.performance_measure = 0;
        self.heading = 0;
        plan = []

        def program(percept):
            stench = percept
            breeze = percept
            glitter = percept

            
            x, y, orientation = update_position(self.x, self.y, self.orientation, self.action)
        
            
            self.KB.tell(expr('%sS_%d%d' % (if_(stench, '', '~'), self.x, self.y)))
            self.KB.tell(expr('%sB_%d%d' % (if_(breeze, '', '~'), self.x, self.y)))
            self.KB.tell(expr('%sGl_%d%d' % (if_(glitter, '', '~'), self.x, self.y)))
            
            
            if glitter: action = 'Grab'
            elif plan: action = plan.pop()
            else:
                for [i, j] in fringe(visited):
                    if self.KB.ask('~P_%d,%d & ~W_%d,%d' % (i, j, i, j)) != False:
                        raise NotImplementedError
                    self.KB.ask('~P_%d,%d | ~W_%d,%d' % (i, j, i, j)) != False 
            if self.action == None: 
                action = random.choice(['Forward', 'TurnRight', 'TurnLeft'])
            return action 

        self.program = program

def update_position(x, y, orientation, action):
    if action == 'TurnRight':
        orientation = turn_right(orientation)
    elif action == 'TurnLeft':
        orientation = turn_left(orientation)
    elif action == 'Forward':
        x, y = x + vector_add((x, y), orientation)
    return x, y, orientation




#_____________________________________________________________________

def unify(x, y, s):
    """Unify expressions x,y with substitution s; return a substitution that
    would make x,y equal, or None if x,y can not unify. x and y can be
    variables (e.g. Expr('x')), constants, lists, or Exprs. [Fig. 9.1]
    >>> ppsubst(unify(x + y, y + C, {}))
    {x: y, y: C}
    """
    if s is None:
        return None
    elif x == y:
        return s
    elif is_variable(x):
        return unify_var(x, y, s)
    elif is_variable(y):
        return unify_var(y, x, s)
    elif isinstance(x, Expr) and isinstance(y, Expr):
        return unify(x.args, y.args, unify(x.op, y.op, s))
    elif isinstance(x, str) or isinstance(y, str):
        return None
    elif issequence(x) and issequence(y) and len(x) == len(y):
        if not x: return s
        return unify(x[1:], y[1:], unify(x[0], y[0], s))
    else:
        return None

def is_variable(x):
    "A variable is an Expr with no args and a lowercase symbol as the op."
    return isinstance(x, Expr) and not x.args and is_var_symbol(x.op)

def unify_var(var, x, s):
    if var in s:
        return unify(s[var], x, s)
    elif occur_check(var, x, s):
        return None
    else:
        return extend(s, var, x)

def occur_check(var, x, s):
    """Return true if variable var occurs anywhere in x
    (or in subst(s, x), if s has a binding for x)."""
    if var == x:
        return True
    elif is_variable(x) and x in s:
        return occur_check(var, s[x], s)
    elif isinstance(x, Expr):
        return (occur_check(var, x.op, s) or
                occur_check(var, x.args, s))
    elif isinstance(x, (list, tuple)):
        return some(lambda element: occur_check(var, element, s), x)
    else:
        return False

def extend(s, var, val):
    """Copy the substitution s and extend it by setting var to val;
    return copy.
    >>> ppsubst(extend({x: 1}, y, 2))
    {x: 1, y: 2}
    """
    s2 = s.copy()
    s2[var] = val
    return s2

def subst(s, x):
    """Substitute the substitution s into the expression x.
    >>> subst({x: 42, y:0}, F(x) + y)
    (F(42) + 0)
    """
    if isinstance(x, list):
        return [subst(s, xi) for xi in x]
    elif isinstance(x, tuple):
        return tuple([subst(s, xi) for xi in x])
    elif not isinstance(x, Expr):
        return x
    elif is_var_symbol(x.op):
        return s.get(x, x)
    else:
        return Expr(x.op, *[subst(s, arg) for arg in x.args])

def fol_fc_ask(KB, alpha):
    """Inefficient forward chaining for first-order logic. [Fig. 9.3]
    KB is a FolKB and alpha must be an atomic sentence."""
    while True:
        new = {}
        for r in KB.clauses:
            ps, q = parse_definite_clause(standardize_variables(r))
            raise NotImplementedError

def standardize_variables(sentence, dic=None):
    """Replace all the variables in sentence with new variables.
    >>> e = expr('F(a, b, c) & G(c, A, 23)')
    >>> len(variables(standardize_variables(e)))
    3
    >>> variables(e).intersection(variables(standardize_variables(e)))
    set([])
    >>> is_variable(standardize_variables(expr('x')))
    True
    """
    if dic is None: dic = {}
    if not isinstance(sentence, Expr):
        return sentence
    elif is_var_symbol(sentence.op):
        if sentence in dic:
            return dic[sentence]
        else:
            v = Expr('v_%d' % standardize_variables.counter.next())
            dic[sentence] = v
            return v
    else:
        return Expr(sentence.op,
                    *[standardize_variables(a, dic) for a in sentence.args])

standardize_variables.counter = itertools.count()

#______________________________________________________________________________

class FOLKB(KB):
    """A knowledge base consisting of first-order definite clauses.
    >>> kb0 = FolKB([expr('Farmer(Mac)'), expr('Rabbit(Pete)'),
    ...              expr('(Rabbit(r) & Farmer(f)) ==> Hates(f, r)')])
    >>> kb0.tell(expr('Rabbit(Flopsie)'))
    >>> kb0.retract(expr('Rabbit(Pete)'))
    >>> kb0.ask(expr('Hates(Mac, x)'))[x]
    Flopsie
    >>> kb0.ask(expr('Wife(Pete, x)'))
    False
    """
    def __init__(self, initial_clauses=[]):
        self.clauses = [] # inefficient: no indexing
        for clause in initial_clauses:
            self.tell(clause)

    def tell(self, sentence):
        if is_definite_clause(sentence):
            self.clauses.append(sentence)
        else:
            raise Exception("Not a definite clause: %s" % sentence)

    def ask_generator(self, query):
        return fol_bc_ask(self, query)

    def retract(self, sentence):
        self.clauses.remove(sentence)

    def fetch_rules_for_goal(self, goal):
        return self.clauses

def test_ask(query, kb=None):
    q = expr(query)
    vars = variables(q)
    answers = fol_bc_ask(kb or test_kb, q)
    return sorted([pretty(dict((x, v) for x, v in a.items() if x in vars))
                   for a in answers],
                  key=repr)

test_kb = FOLKB(
    map(expr, ['Farmer(Mac)',
               'Rabbit(Pete)',
               'Mother(MrsMac, Mac)',
               'Mother(MrsRabbit, Pete)',
               '(Rabbit(r) & Farmer(f)) ==> Hates(f, r)',
               '(Mother(m, c)) ==> Loves(m, c)',
               '(Mother(m, r) & Rabbit(r)) ==> Rabbit(m)',
               '(Farmer(f)) ==> Human(f)',
               # Note that this order of conjuncts
               # would result in infinite recursion:
               #'(Human(h) & Mother(m, h)) ==> Human(m)'
               '(Mother(m, h) & Human(h)) ==> Human(m)'
               ])
)

crime_kb = FOLKB(
  map(expr,
    ['(American(x) & Weapon(y) & Sells(x, y, z) & Hostile(z)) ==> Criminal(x)',
     'Owns(Nono, M1)',
     'Missile(M1)',
     '(Missile(x) & Owns(Nono, x)) ==> Sells(West, x, Nono)',
     'Missile(x) ==> Weapon(x)',
     'Enemy(x, America) ==> Hostile(x)',
     'American(West)',
     'Enemy(Nono, America)'
     ])
)

def fol_bc_ask(KB, query):
    """A simple backward-chaining algorithm for first-order logic. [Fig. 9.6]
    KB should be an instance of FolKB, and goals a list of literals.
    >>> test_ask('Farmer(x)')
    ['{x: Mac}']
    >>> test_ask('Human(x)')
    ['{x: Mac}', '{x: MrsMac}']
    >>> test_ask('Hates(x, y)')
    ['{x: Mac, y: MrsRabbit}', '{x: Mac, y: Pete}']
    >>> test_ask('Loves(x, y)')
    ['{x: MrsMac, y: Mac}', '{x: MrsRabbit, y: Pete}']
    >>> test_ask('Rabbit(x)')
    ['{x: MrsRabbit}', '{x: Pete}']
    >>> test_ask('Criminal(x)', crime_kb)
    ['{x: West}']
    """
    return fol_bc_or(KB, query, {})

def fol_bc_or(KB, goal, theta):
    for rule in KB.fetch_rules_for_goal(goal):
        lhs, rhs = parse_definite_clause(standardize_variables(rule))
        for theta1 in fol_bc_and(KB, lhs, unify(rhs, goal, theta)):
            yield theta1

def fol_bc_and(KB, goals, theta):
    if theta is None:
        pass
    elif not goals:
        yield theta
    else:
        first, rest = goals[0], goals[1:]
        for theta1 in fol_bc_or(KB, subst(theta, first), theta):
            for theta2 in fol_bc_and(KB, rest, theta1):
                yield theta2

#______________________________________________________________________________

# Example application (not in the book).
# You can use the Expr class to do symbolic differentiation.  This used to be
# a part of AI; now it is considered a separate field, Symbolic Algebra.

def diff(y, x):
    """Return the symbolic derivative, dy/dx, as an Expr.
    However, you probably want to simplify the results with simp.
    >>> diff(x * x, x)
    ((x * 1) + (x * 1))
    >>> simp(diff(x * x, x))
    (2 * x)
    """
    if y == x: return ONE
    elif not y.args: return ZERO
    else:
        u, op, v = y.args[0], y.op, y.args[-1]
        if op == '+': return diff(u, x) + diff(v, x)
        elif op == '-' and len(args) == 1: return -diff(u, x)
        elif op == '-': return diff(u, x) - diff(v, x)
        elif op == '*': return u * diff(v, x) + v * diff(u, x)
        elif op == '/': return (v*diff(u, x) - u*diff(v, x)) / (v * v)
        elif op == '**' and isnumber(x.op):
            return (v * u ** (v - 1) * diff(u, x))
        elif op == '**': return (v * u ** (v - 1) * diff(u, x)
                                 + u ** v * Expr('log')(u) * diff(v, x))
        elif op == 'log': return diff(u, x) / u
        else: raise ValueError("Unknown op: %s in diff(%s, %s)" % (op, y, x))

def simp(x):
    if not x.args: return x
    args = map(simp, x.args)
    u, op, v = args[0], x.op, args[-1]
    if op == '+':
        if v == ZERO: return u
        if u == ZERO: return v
        if u == v: return TWO * u
        if u == -v or v == -u: return ZERO
    elif op == '-' and len(args) == 1:
        if u.op == '-' and len(u.args) == 1: return u.args[0] ## --y ==> y
    elif op == '-':
        if v == ZERO: return u
        if u == ZERO: return -v
        if u == v: return ZERO
        if u == -v or v == -u: return ZERO
    elif op == '*':
        if u == ZERO or v == ZERO: return ZERO
        if u == ONE: return v
        if v == ONE: return u
        if u == v: return u ** 2
    elif op == '/':
        if u == ZERO: return ZERO
        if v == ZERO: return Expr('Undefined')
        if u == v: return ONE
        if u == -v or v == -u: return ZERO
    elif op == '**':
        if u == ZERO: return ZERO
        if v == ZERO: return ONE
        if u == ONE: return ONE
        if v == ONE: return u
    elif op == 'log':
        if u == ONE: return ZERO
    else: raise ValueError("Unknown op: " + op)
    ## If we fall through to here, we can not simplify further
    return Expr(op, *args)

def d(y, x):
    "Differentiate and then simplify."
    return simp(diff(y, x))

#_______________________________________________________________________________

# Utilities for doctest cases
# These functions print their arguments in a standard order
# to compensate for the random order in the standard representation

def pretty(x):
    t = type(x)
    if t is dict:  return pretty_dict(x)
    elif t is set: return pretty_set(x)
    else:          return repr(x)

def pretty_dict(d):
    """Return dictionary d's repr but with the items sorted.
    >>> pretty_dict({'m': 'M', 'a': 'A', 'r': 'R', 'k': 'K'})
    "{'a': 'A', 'k': 'K', 'm': 'M', 'r': 'R'}"
    >>> pretty_dict({z: C, y: B, x: A})
    '{x: A, y: B, z: C}'
    """
    return '{%s}' % ', '.join('%r: %r' % (k, v)
                              for k, v in sorted(d.items(), key=repr))

def pretty_set(s):
    """Return set s's repr but with the items sorted.
    >>> pretty_set(set(['A', 'Q', 'F', 'K', 'Y', 'B']))
    "set(['A', 'B', 'F', 'K', 'Q', 'Y'])"
    >>> pretty_set(set([z, y, x]))
    'set([x, y, z])'
    """
    return 'set(%r)' % sorted(s, key=repr)

def pp(x):
    print pretty(x)

def ppsubst(s):
    """Pretty-print substitution s"""
    ppdict(s)

def ppdict(d):
    print pretty_dict(d)

def ppset(s):
    print pretty_set(s)

#________________________________________________________________________
# CTM: misc. helpers to extend the interface

def is_literal(thing):
    return isinstance(thing,Expr) and \
           ((thing.op == '~' and len(things.args) == 1) \
            or (len(thing.args) == 0))
    
def is_literal_positive(literal):
    return not literal.op == '~'

def literal_name(literal):
    if literal.op == '~':
        return literal.args[0].op
    else:
        return literal.op

def clauses_to_conjunct(clause_list):
    """ coerce a list of clauses into a conjunction """
    conj = Expr('&')
    conj.args = clause_list
    return conj
    #return ' & '.join(map(lambda(i): '{0}'.format(KB.clauses[i]), list))

def prop_symbols_from_KB(kb):
    """ CTM: This is very inefficient,
    but I can't figure out why direct list iteration doesn't work """
    return prop_symbols(clauses_to_conjunct(kb.clauses))

def prop_symbols_from_clause_list(clause_list):
    return prop_symbols(clauses_to_conjunct(clause_list))

#________________________________________________________________________





#-------------------------------------------------------------------------------

if __name__ == '__main__':
    """
    The main funciton called when wumpus_test.py is run from the command line:
    > python wumpus_test.py <options>
    """
print('\n--------------------1. Create WumpusEnvironment---------------------\n')
    #Creates a Wumpus World Scenario
r=wscenario_5x5()
    #options = readCommand( sys.argv[1:] )
    #run_command( options )




#env = WumpusEnvironment()
#print env.to_string()

wEnv = WumpusEnvironment(5,5)
wEnv.add_thing(Wumpus(),(1,3))
wEnv.add_thing(Pit(),(3,3))
wEnv.add_thing(Pit(),(3,1))
wEnv.add_thing(Gold(),(2,3))
wEnv.add_thing(Pit(),(4,4))
explorer = Explorer(heading='north', verbose=True)
explorer.heading = 0 #North
wEnv.add_thing(explorer, (2,2))
print 'Explorer Location: ', explorer.location
plWumpus = PLWumpusAgent()

#explorer = W.Explorer(heading='north', verbose=True)

#print 'Step made'
#print wEnv.to_string()
print '-------------------------#2&3 Percept, Sense the Environment---------\n'
percept2 = wEnv.percept(explorer)

print wEnv.percept(explorer)
pvec = explorer.raw_percepts_to_percept_vector(percept2)
senses = [explorer.pretty_percept_vector(pvec)[5], explorer.pretty_percept_vector(pvec)[6], explorer.pretty_percept_vector(pvec)[7], explorer.pretty_percept_vector(pvec)[8], explorer.pretty_percept_vector(pvec)[9]]
print senses;

print('\n-------------------------Manual Simulation----------------\n')
print wEnv.to_string()

print '\n------------------------------ --------------------------\n'
while(True):
    print 'Ending actions: [quit, stop, exit] Possible actions = [TurnRight, TurnLeft, Forward, Grab, Release, Shoot, Wait] '
    wEnv.step()
    print wEnv.percept(explorer)
    print wEnv.to_string()
    print 'Sense the environment', wEnv.percept(explorer)
    percept2 = wEnv.percept(explorer)
    pvec = explorer.raw_percepts_to_percept_vector(percept2)
    senses = [explorer.pretty_percept_vector(pvec)[5], explorer.pretty_percept_vector(pvec)[6], explorer.pretty_percept_vector(pvec)[7], explorer.pretty_percept_vector(pvec)[8], explorer.pretty_percept_vector(pvec)[9]]   
    
    print 'Environment', senses;
    print '\n------------------------------  --------------------------\n'
    
    n = raw_input("To quit the Simulation, enter 'q':")
    if n.strip() == 'q':
        break

#try:
    #print [method for method in dir(wEnv) if callable(getattr(wEnv, method))]
#except:
    #raise

#wEnv.is_done()
print 'Sense the environment', wEnv.percept(explorer)
print wEnv.percept(explorer)
percept2 = wEnv.percept(explorer)
pvec = explorer.raw_percepts_to_percept_vector(percept2)
senses = [explorer.pretty_percept_vector(pvec)[5], explorer.pretty_percept_vector(pvec)[6], explorer.pretty_percept_vector(pvec)[7], explorer.pretty_percept_vector(pvec)[8], explorer.pretty_percept_vector(pvec)[9]]   
print 'Environment', senses;




print ('\n-------------------------4. Create PLWumpusAgent----------------\n')

#Creat the Wumpus Agent
plWumpus = PLWumpusAgent()
print plWumpus
percept  = 'stench'
print ('PLWumpus Agent percept: %s' % percept) 
plWumpus.program(percept)
print plWumpus.program(percept)
print plWumpus.KB.clauses
print 'PLWumpus location: ', plWumpus.x, plWumpus.y


print ('\n-------------------------5. Test Agent----------------\n')
steps = 0
steps = raw_input("\nEnter in the number of steps you want to run ")
steps = steps.strip()
print 'Number of Steps: %s' % steps
steps = int(steps)
#test_agent(explorer, steps, wEnv)
e = [wEnv] 
plWumpus.program(senses[2])
performance = test_agent(plWumpus, 3, wEnv, senses)
print performance
print '-------------------------------End --------------------------------'