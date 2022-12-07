//#include <iomanip>

volScalarField zoneIDfield
(
    IOobject
    (
        "zoneIDfield",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("zero",dimensionSet(0,0,0,0,0,0,0),0.0)
);

Info << "start" << nl << endl;
// Create fluid Cell ID file
std::ofstream zid;
zid.open("fluidID.csv", std::ofstream::out | std::ofstream::app);


//Create Solid Cell ID File
std::ofstream zid2;
zid2.open("solidID.csv", std::ofstream::out | std::ofstream::app);


label f1i = 0;
label f2i = 0;

forAll(mesh.C(), celli)
{
  cout <<std:: fixed;
    if(gamma[celli] == 0.0)
    {
        zid << celli << nl;
        f1i += 1;
        if(f1i % 1000 == 0)
        {Info << "fluid" << f1i << nl << endl;}
    }
    else
    {
        zid2 << celli << nl;
        f2i += 1;
        if(f2i % 1000 == 0)
        {Info << "solid" << f2i << nl << endl;}
    }
}
zid.close();
zid2.close();


Info << "fluid" << f1i << nl << endl;
Info << "solid" << f2i << nl << endl;

forAll(mesh.C(), celli)
{
        zoneIDfield[celli] = celli;
}

runTime.setTime(100, 100);
runTime.write(); 
zoneIDfield.write();
Info << "Done write cell ID" << nl << endl;

