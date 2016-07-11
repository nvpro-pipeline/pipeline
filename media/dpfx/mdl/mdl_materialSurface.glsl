_materialSurface mdl_materialSurface( in vec4 scattering, in _materialEmission emission )
{
  _materialSurface ms;
  ms.scattering = scattering;
  ms.emission   = emission;
  return( ms );
}

