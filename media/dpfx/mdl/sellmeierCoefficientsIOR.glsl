vec3 sellmeierCoefficientsIOR( in vec3 sellmeierB, in vec3 sellmeierC )
{
  float l2 = 0.5892 * 0.5892; // simplified to fixed wavelength of 589.2 nm
  return( vec3( sqrt( 1.0 + sellmeierB.x * l2 / ( l2 - sellmeierC.x ) + sellmeierB.y * l2 / ( l2 - sellmeierC.y ) + sellmeierB.z * l2 / ( l2 - sellmeierC.z ) ) ) );
}

