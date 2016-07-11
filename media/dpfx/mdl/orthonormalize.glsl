vec3 orthonormalize( in vec3 v0, in vec3 v1 )
{
  //  determine the orthogonal projection of v1 on v0 : ( v0 * v1 ) * v0
  //  and subtract it from v1 resulting in the orthogonalized version of v1
  return( normalize( v1 - dot( v0, v1 ) * v0 ) );
}

