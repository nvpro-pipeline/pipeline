
void main(void)
{
  normal    = normalize( varNormal );
  texCoord0 = varTexCoord0;
  evalTemporaries( normal );
  float alphaCutout = evalAlphaCutout();
  Color = vec4( texCoord0.xy, 0, 1 );
  //if ( alphaCutout == 0.0f )
  //{
  //  //discard;
  //  Color = vec4(1,0,0,1);
  //}
  //else
  //{
  //  Color = vec4(1,1,1,1);
  //}
}
