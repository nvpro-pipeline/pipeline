float luminance( in vec3 v )
{
  // Luma coefficients according to ITU-R Recommendation BT.709 (http://en.wikipedia.org/wiki/Rec._709)
  return( 0.2126f * v.r + 0.7152f * v.g + 0.0722f * v.b );
}

void main(void)
{
  vec4 materialTransparent = evaluateColor( transparentColor, transparentSampler, transparentTC );
  float a = ( rgbTransparency ? 1.0f - luminance( materialTransparent.rgb ) : materialTransparent.a ) * transparency;

  if ( a < 1.0f )
  {
    discard;
  }
  
  emitColor( vec4( 1 ));
}
