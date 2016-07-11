vec3 mdl_base_checkerBumpTexture( in _base_textureCoordinateInfo uvw, in float factor, in float blur, in float checkerPosition, in vec3 normal )
{
  const float delta = 0.025f;   // magic, looks good with this value
  const vec3 black = vec3( 0.0f, 0.0f, 0.0f );
  const vec3 white = vec3( 1.0f, 1.0f, 1.0f );

  _base_textureCoordinateInfo uvwLocal;
  uvwLocal.position = uvw.position + delta * vec3( 0.0f, 0.0f, 0.0f );
  float r0 = mdl_math_luminance( mdl_base_checkerTexture( uvwLocal, black, white, blur, checkerPosition ).tint );

  uvwLocal.position = uvw.position + delta * uvw.tangentU;
  float r1 = mdl_math_luminance( mdl_base_checkerTexture( uvwLocal, black, white, blur, checkerPosition ).tint );

  uvwLocal.position = uvw.position + delta * uvw.tangentV;
  float r2 = mdl_math_luminance( mdl_base_checkerTexture( uvwLocal, black, white, blur, checkerPosition ).tint );

  uvwLocal.position = uvw.position + delta * normal;
  float r3 = mdl_math_luminance( mdl_base_checkerTexture( uvwLocal, black, white, blur, checkerPosition ).tint );

  return( normalize( normal - ( r1 - r0 ) * uvw.tangentU - ( r2 - r0 ) * uvw.tangentV - ( r3 - r0 ) * normal ) );
}

