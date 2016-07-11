_materialEmission mdl_materialEmission( in vec4 emission, in vec3 intensity, int mode )
{
  _materialEmission mi;
  mi.emission   = emission;
  mi.intensity  = intensity;
  return( mi );
}

