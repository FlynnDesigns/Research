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
std::vector<long int> trueValues;
//double minX = 0;
//double minY = 0;
//bool foundValue = false;
std::ifstream file("solid_coordinates.txt");
int counter = 0;

// read the input into a vector
std::copy(std::istream_iterator<double>(file), std::istream_iterator<double>(), std::back_inserter(values));
std::cout << "The size of the vector is = " << values.size() << std::endl;

// Storing text file into x and y arrays
for (size_t i=0; i < values.size();i+=2)
{
    x.push_back(values[i]);
    y.push_back(values[i+1]);
};

std::cout << "The size of the x vector is = " << x.size() << std::endl;
std::cout << "The size of the y vector is = " << y.size() << std::endl;

//minX = *min_element(x.begin(), x.end());
//minY = *min_element(y.begin(), y.begin());

/*
for (int i = 0; i < y.size(); ++i)
{
	std::cout << y[i] << std::endl;
};
*/

// Using x and y vectors to tell openfoam which cells are solid
forAll(mesh.C(), celli)
{
	//std::cout << mesh.C()[celli].component(0) << " , " << mesh.C()[celli].component(1) << std::endl;
	// Checking to see if the y component is in bounds
	if( mesh.C()[celli].component(1) >= 5.5 && mesh.C()[celli].component(1) <= 68.5 )
	{
		// Checking to see if the x component matches
		if( mesh.C()[celli].component(0) >= 1.5 && mesh.C()[celli].component(1) <= 64.5 )
		{
			// Looping through the list of values of x and y 
			for (size_t i = 0; i < x.size();++i)
			{
				// Looping through elements in x and y of your geometry that you feed in 
				if (mesh.C()[celli].component(1) == y[i] && mesh.C()[celli].component(0) == x[i])
			    	{	
			    		trueValues.push_back(celli);
			    		counter += 1;
					//std::cout << "Value found at ("<< celli << ")" << std::endl;
					break;
				};
			};
		};
	}
};

// Sorting the found celli values in assending order
std::sort(trueValues.begin(), trueValues.end());

// Making sure that the gamma is 0
forAll(mesh.C(), celli)
{
	gamma[celli] = 0;
};


forAll(mesh.C(), celli)
{
	if (std::find(trueValues.begin(), trueValues.end(), celli) != trueValues.end())
	{
		gamma[celli] = 0.5;
		//std::cout << "Confirm (" << celli << ")" << std::endl;
	}
};


std::cout << "counter = " << counter << std::endl; 
// boundary patch of gamma equals to the neigbour cell value
forAll(gamma.boundaryField(), patchi)
{
    gamma.boundaryFieldRef()[patchi] == gamma.boundaryField()[patchi].patchInternalField();
}

