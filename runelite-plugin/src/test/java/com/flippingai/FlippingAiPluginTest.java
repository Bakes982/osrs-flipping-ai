package com.flippingai;

import net.runelite.client.RuneLite;
import net.runelite.client.externalplugins.ExternalPluginManager;

public class FlippingAiPluginTest
{
    public static void main(String[] args) throws Exception
    {
        ExternalPluginManager.loadBuiltin(FlippingAiPlugin.class);
        RuneLite.main(args);
    }
}
