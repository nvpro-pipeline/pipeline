// look here: http://en.wikipedia.org/wiki/Fresnel_equations
float fresnel( in float eta1, in float eta2, in float cosTheta1 )
{
  float etaInv = eta1 / eta2;
  float cosTheta2 = 1.0f - ( 1.0f - cosTheta1 * cosTheta1 ) * ( etaInv * etaInv );
  if ( 0.0f <= cosTheta2 )
  {
    cosTheta2 = sqrt( cosTheta2 );
    float n1t1 = eta1 * cosTheta1;
    float n1t2 = eta1 * cosTheta2;
    float n2t1 = eta2 * cosTheta1;
    float n2t2 = eta2 * cosTheta2;
    float rs = ( n1t1 - n2t2 ) / ( n1t1 + n2t2 );
    float rp = ( n1t2 - n2t1 ) / ( n1t2 + n2t1 );
    float f = 0.5f * ( rs * rs + rp * rp );
    return( clamp( f, 0.0f, 1.0f ) );
  }
  else
  {
    return( 1.0f );
  }
}

// we assume, light rays run through air with ior == 1.0f
// with eta = eta2 / eta1, we have
//  - when hitting a front face: eta2 == ior, eta1 == 1.0f => eta = ior
//  - when hitting a back face : eta2 == 1.0f, eta1 == ior => eta = 1.0f / ior
vec3 fresnel( in vec3 N, in vec3 ior )
{
  float cosTheta1 = dot( N, viewDir );
  if ( gl_FrontFacing )
  {
    return( vec3( fresnel( 1.0f, ior[0], cosTheta1 ), fresnel( 1.0f, ior[1], cosTheta1 ), fresnel( 1.0f, ior[2], cosTheta1 ) ) );
  }
  else
  {
    return( vec3( fresnel( ior[0], 1.0f, cosTheta1 ), fresnel( ior[1], 1.0f, cosTheta1 ), fresnel( ior[2], 1.0f, cosTheta1 ) ) );
  }
}

vec4 mdl_df_fresnelLayer( in vec3 ior, in float weight, in vec4 layer, in vec4 base, in vec3 normal )
{
  return( vec4( mix( base.rgb, layer.rgb, weight * fresnel( normal, ior ) ), base.a ) );
}

