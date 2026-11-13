## Use of AI assistance

Claude AI was given the stochastic policy training code and the ESP32 code repository, and prompted as follows:

> I trained a neural corrector for a custom LQR and implemented it on a esp32.
> I want now to compare it to another controller.
> I want you to give me code to obtain another controller that corrects the LQR like mine (use the ensemble models as dynamics as I am doing). It must be implementable on a ESP32.
> It doesn't have to be trained, I don't care, just give me another controller that can substitute mine in the framework.

Following the first results, it was prompted again to refine the policy:

> I want you to improve the performance. It works but there are still a lot of oscillations. Keep the general structure but give me code that increases performance.