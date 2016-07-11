_base_anisotropyReturn mdl_base_anisotropyConversion( in float roughness, in float anisotropy, in float anisotropyRotation, in vec3 tangentU, bool miaAnisotropySemantic )
{
  _base_anisotropyReturn aniso;
  aniso.roughnessU = roughness;
  aniso.roughnessV = roughness * anisotropy;
  vec3 tangentV = normalize( cross( stateNormal, tangentU ) );
  float angle = 2.0f * PI * anisotropyRotation;
  aniso.tangentU = cos( angle ) * tangentU + sin( angle ) * tangentV;
  return( aniso );
}

