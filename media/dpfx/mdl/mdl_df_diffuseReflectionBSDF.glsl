vec4 mdl_df_diffuseReflectionBSDF( in vec3 materialDiffuse, in float roughness, in vec3 normal )
{
  float cosThetaI = max( 0.0f, dot( normal, lightDir ) );
  float factor = cosThetaI;
  if ( ( 0.0f < factor ) && ( 0.0f < roughness ) )
  {
    // see http://en.wikipedia.org/wiki/Oren%E2%80%93Nayar_reflectance_model
    float sigmaSquare = 0.25f * PI_SQUARE * roughness * roughness;
    float A = 1.0f - 0.5f * sigmaSquare / ( sigmaSquare + 0.33f );
    float B = 0.45f * sigmaSquare / ( sigmaSquare + 0.09f );

    // project lightDir and viewDir on surface to get the azimuthal angle between them
    // as we don't really need the projections, but the angle between them, it's enough to just use the cross instead
    vec3 pl = normalize( cross( lightDir, normal ) );
    vec3 pv = normalize( cross( viewDir, normal ) );
    float cosPhi = max( 0.0f, dot( pl, pv ) );

    float sinAlpha, tanBeta;
    float cosThetaO = max( 0.0f, dot( normal, viewDir ) );
    float sinThetaI = sqrt( max( 0.0f, 1.0f - cosThetaI * cosThetaI ) );
    float sinThetaO = sqrt( max( 0.0f, 1.0f - cosThetaO * cosThetaO ) );
    if ( cosThetaI < cosThetaO )
    { // -> thetaO < thetaI
      sinAlpha = sinThetaI;
      tanBeta = sinThetaO / cosThetaO;
    }
    else
    {
      sinAlpha = sinThetaO;
      tanBeta = sinThetaI / cosThetaI;
    }

    factor *= A + B * cosPhi * sinAlpha * tanBeta;
  }

  return( vec4( factor * lightDiffuse * materialDiffuse, 1.0f ) );
}

vec4 mdl_df_diffuseReflectionBSDFEnvironment( in vec3 tint, in float roughness, in vec3 normal )
{
  vec3 rgb = vec3( 0.0f, 0.0f, 0.0f );
  if ( sys_EnvironmentSamplerEnabled )
  {
    lightDir = reflect( -viewDir, normal );
    lightDiffuse = evalEnvironmentMap( lightDir, roughness );
    rgb = mdl_df_diffuseReflectionBSDF( tint, roughness, normal ).rgb;
  }
  return( vec4( rgb, 1.0f ) );
}
