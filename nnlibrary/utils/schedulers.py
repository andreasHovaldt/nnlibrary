import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import weakref

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from nnlibrary.engines.train import Trainer

class OneCycleLR(lr_scheduler.OneCycleLR):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        trainer: "Trainer | None" = None,
        max_lr: float | list[float] = 0.001, 
        total_steps: int | None = None, 
        epochs: int | None = None, 
        steps_per_epoch: int | None = None, 
        pct_start: float = 0.3, 
        anneal_strategy: Literal['cos'] | Literal['linear'] = "cos", 
        cycle_momentum: bool = True, 
        base_momentum: float | list[float] = 0.85, 
        max_momentum: float | list[float] = 0.95, 
        div_factor: float = 25, 
        final_div_factor: float = 10000, 
        three_phase: bool = False, 
        last_epoch: int = -1,
    ) -> None:
        
        # Use trainer weak proxy to get actual training steps if available
        if (total_steps is not None) or ((epochs is not None) and (steps_per_epoch is not None)):
            pass
        
        elif trainer is not None:
            try:
                total_steps = trainer.num_epochs * len(trainer.trainloader)
            except ReferenceError:
                raise ReferenceError("The trainer proxy is dead and has probably been garbage collected!")
        
        super().__init__(
            optimizer=optimizer, 
            max_lr=max_lr, 
            total_steps=total_steps, 
            epochs=epochs, 
            steps_per_epoch=steps_per_epoch, 
            pct_start=pct_start, 
            anneal_strategy=anneal_strategy, 
            cycle_momentum=cycle_momentum, 
            base_momentum=base_momentum, 
            max_momentum=max_momentum, 
            div_factor=div_factor, 
            final_div_factor=final_div_factor, 
            three_phase=three_phase, 
            last_epoch=last_epoch,
        )