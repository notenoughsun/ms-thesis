# Structure plan

Presentation content overall

25-30 slides

15 - work, other is for responces to experts questions

1. Introduction
2. Problem
3. Most important characteristics of LNS
4. Data collection and processing mb, the research of a landscape and all, include diagrams here????
5. the logic explanation
6. system architecture diagram
    1. what is the logic of what i'm doing?? what questions here i have to explain
7. general final system representation
    1. say about physical implementation, ARcore and alll
    2. dataset collection and experiments
        1. plan experiments, what i'm testing here??
            1. use visual odometry from ar, measure the trajectrories over "T" shaped corridor, solve global optimization and create the map
            2. check for the ways of magnetic field collection
                1. how to exploit the bilinear structure of the magnetic field?? write formulas here and explain the logic more

- [ ]  introduction
- [ ]  literature review
- [ ]  motivation
state of the art systems
system design & architecture
- [ ]  background, what is system, what features, how does it works
data processing
- [ ]  slam and graph minimization
factor graphs, distance function, graph matching
trajectory generation
walking model, transfer function / motion model (visual odometry)
- [ ]  map construction / reconstruction
observations to map (approximation, regression), map to observations (as a sensor)
(sensor model), propagation model, slam / sam
- [ ]  conditioning on magnetic field data
what graphs / figures can be plotted (prove the convergence)

> Write the FOMs here, explain the criterias

> State the Goal here: create a model with these characteristics, which is based on ... (slide 3)

// Preprocessing of input image

## exploitation of the bilinear structure of the magnetic field

write formulas here and explain the logic more

the field we measure and utilize is the usual magnetic field of the earth. because of the metal constructions affecting the field, inside the buildings there exists a special pattern of magnetic field orientation. 

To say a word about magnetic field characteristics, here we state, that this field can be detected and measured in several locations with the sensors of usual on-the-shelf smartphones. The amplitude of any signal in a magnetic field is decreasing quadratically with the distance. 

What do we need from this magnetic field? We aim to use the information from this field for localization. Once we have a match between observations and the map, we can say, that there is a match between the user location and a point on a map and return the calculated location. 

This algorithm can be represented as image to image mapping as in puzzle games. We have a small image given by our latest observations, and a map represented by a larger image.

 

Because of this fact, we can better measure the signal near the walls, and we also have no information 

is the usu

1.