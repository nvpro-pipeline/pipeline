mat4 mdl_base_rotationTranslationScale( in vec3 rotation, in vec3 translation, in vec3 scaling )
{
  mat4 st = mat4( scaling.x, 0.0f, 0.0f, 0.0f
                , 0.0f, scaling.y, 0.0f, 0.0f
                , 0.0f, 0.0f, scaling.z, 0.0f
                , translation.x - 0.5f, translation.y - 0.5f, translation.z - 0.5f, 1.0f );
  vec3 s = sin( rotation );
  vec3 c = cos( rotation );
  mat4 r = mat4( c.y * c.z, -c.x * s.z + s.x * s.y * c.z,  s.x * s.z + c.x * s.y * c.z, 0.0f
               , c.y * s.z,  c.x * c.z + s.x * s.y * s.z, -s.x * c.z + c.x * s.y * s.z, 0.0f
               , -s.y     ,  s.x * c.y                  ,  c.x * c.y                  , 0.0f
               , 0.5f     ,  0.5f                       ,  0.5f                       , 1.0f );
  return( st * r );
}

