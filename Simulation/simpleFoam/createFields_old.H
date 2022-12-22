#include "createMRF.H"
#include "createFvOptions.H"

// Create a variable: Density Field Gamma
volScalarField gamma
(
    IOobject
    (
        "gamma",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("zero",dimensionSet(0,0,0,0,0,0,0),1.0)
);

std::vector<double> values;
std::vector<double> x;
std::vector<double> y;
double minX = 0;
double minY = 0;
bool foundValue = false;
std::ifstream file("solid_coordinates.txt");

// read the input
std::copy(std::istream_iterator<double>(file), std::istream_iterator<double>(), std::back_inserter(values));
// Storing text file into x and y arrays
for (size_t i=0; i < values.size();i+=2)
{
    x.push_back(values[i]);
    y.push_back(values[i+1]);
    std::cout <<values[i] << " , " << values[i+1] << std::endl;
};

minX = *min_element(x.begin(), x.end());
minY = *min_element(y.begin(), y.begin());

// Using x and y vectors to tell openfoam which cells are solid
forAll(mesh.C(), celli)
{
if( mesh.C()[celli].component(1) >= minY && mesh.C()[celli].component(0) >= minX)
{
	for (size_t i = 0;i < values.size();++i)
	{
	    	//std::cout << "In loop" << std::endl;
	    	if (mesh.C()[celli].component(1) == y[i] && mesh.C()[celli].component(0) == x[i])
	    	{
				gamma[celli] = 0.5;
				// Line below is used for debugging
				std::cout << "Value at: " <<x[i] << " , " << y[i]  << std::endl;
				foundValue = true;
				//break;
	    	};
	  
	};

	if (foundValue == false)
	{
	    gamma[celli] = 0.0;
	    std::cout << "Value not found" << std::endl;
	};

	foundValue = false;	
}
else
{
gamma[celli] = 0.0;
}
}

// boundary patch of gamma equals to the neigbour cell value
forAll(gamma.boundaryField(), patchi)
{
    gamma.boundaryFieldRef()[patchi] == gamma.boundaryField()[patchi].patchInternalField();
}

