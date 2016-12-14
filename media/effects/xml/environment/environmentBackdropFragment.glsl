#if 0

// DAR Guessing: This path seem to fail when the sampler2D is defined but not used. 
// It will be reported as unreferenced by the GLSL compiler then and that might trigger a nullptr in location here:
// 
// ParameterCache<ParameterCacheStream>::updateContainer( ContainerGLHandle container )
// ==> CRASH: unsigned char *basePtr = &m_uniformData[0] + location.m_offset; 

void main(void)
{
  Color = vec4( normalize( varTexCoord0 ) * 0.5f + 0.5f, 1.0f ); // Visualize the direction vectors used as texture lookup.
}

#else

#ifndef M_PI
#define M_PI 3.14159265358979f
#endif

// Spherical environment map backdrop.
void main(void)
{
  vec3  dir = normalize( varTexCoord0 ); // varTexCoord0 contains the vectors from the camera position to the corners of the view frustum.
  float phi = ( atan( dir.x, -dir.z ) + M_PI ) / ( 2.0f * M_PI ); // Map that direction onto the spherical environment texture longitude.
  Color = texture( environment, vec2( phi, acos( -dir.y ) / M_PI ) );
}

#endif
